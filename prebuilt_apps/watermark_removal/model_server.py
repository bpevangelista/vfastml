import argparse
import asyncio
import io
from abc import ABC

import boto3
import joblib
import numpy as np
import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity

from vfastml import log
from vfastml.engine.dispatch_requests import ImageToImageReq, DispatchRequestResult
from vfastml.models_servers.model_server_image import ImageToImageModelServer
from vfastml.utils import print_model_parameters
from vfastml.utils.network import download_uris_async

s3 = boto3.client("s3")
BUCKET_NAME = ''
PROJECT_NAME = ''


def resize_image(image_tensor, min_dimension=768, max_dimension=1024):
    _, height, width = image_tensor.shape

    # noinspection PyTypeChecker
    scale_factor = torch.where(height > width, min_dimension / height, max_dimension / width)
    new_height = torch.round(height * scale_factor).to(torch.int32)
    new_width = torch.round(width * scale_factor).to(torch.int32)

    if new_width == width and new_height == height:
        return image_tensor

    resized_image = F.interpolate(image_tensor.unsqueeze(0),
                                  size=(new_height, new_width),
                                  mode='bilinear')

    return resized_image


def download_from_s3(bucket: str, object_path: str) -> io.BytesIO | None:
    try:
        file_data = io.BytesIO()
        s3.download_fileobj(bucket, object_path, file_data)
        file_data.seek(0)
        return file_data
    except Exception as e:
        print(e)
        return None


def upload_to_s3(data: bytes, bucket: str, object_path: str) -> str | None:
    try:
        s3.put_object(Body=data, Bucket=bucket, Key=object_path)
    except Exception as e:
        print(e)
        return None
    return f's3://{bucket}/{object_path}'


def pil_image_to_tensor(image: Image):
    # Only support 1bpp or 8bpp
    image_tensor = torch.from_numpy(np.array(image, np.uint8, copy=True))
    if image.mode == '1':
        image_tensor *= 255

    image_channels = len(image.getbands())
    image_tensor = image_tensor.view(image.height, image.width, image_channels)
    # [height, width, channels] -> [channels, height, width]
    image_tensor = image_tensor.permute(2, 0, 1).contiguous()

    return image_tensor.float().div(255.0).unsqueeze(dim=0)


class ImageToImageModelCleanupServer(ImageToImageModelServer, ABC):
    async def _image_to_image(self, requests: list[ImageToImageReq]):
        for request in requests:
            images_futures = download_uris_async(request.images)

            src_images_uri: list[str] = []
            result_images_uri: list[str] = []
            for image_future in asyncio.as_completed(images_futures):
                image_uri, image_data = await image_future
                if image_data is None:
                    continue

                image = Image.open(io.BytesIO(image_data))
                image_tensor = pil_image_to_tensor(image)

                # Any image pre-processing

                with profile(activities=[ProfilerActivity.CPU], profile_memory=False, record_shapes=True) as prof:
                    with record_function("model"):
                        with torch.no_grad():
                            predicted = self.model(image_tensor, **request.forward_params)

                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

                src_images_uri.append(upload_to_s3(image_tensor, BUCKET_NAME, f'{image_uri}_original.png'))
                result_images_uri.append(upload_to_s3(predicted, BUCKET_NAME, f'{image_uri}_predicted.png'))

            result = {
                'images_uri': src_images_uri,
                'cleaned_images_uri': result_images_uri,
            }

            dispatch_result = DispatchRequestResult(request.request_id, result=result)
            await self._results_queue.put(dispatch_result)


    async def _load_model(self):
        log.info(f'Loading Model {self.model_uri}')

        if self.model_uri.startswith('s3://'):
            bucket, file_path = self.model_uri.split('//')[1].split('/', 1)
            file_data = download_from_s3(bucket, file_path)
            self.model = joblib.load(file_data)
        else:
            self.model = joblib.load(self.model_uri)

        # Handle float denorm
        print_model_parameters(self.model)
        eps = 1e-5
        for param in self.model.parameters():
            # noinspection PyTypeChecker
            param.data = torch.where(torch.abs(param) < eps, torch.zeros_like(param), param)

        # Config
        # torch.set_num_threads(16)
        self.model = self.model.half().to('cuda:0')

        self.model.eval()


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log-level', type=str, default='debug')
    return parser.parse_args()


if __name__ == '__main__':
    args = handle_args()

    model_server = ImageToImageModelCleanupServer(
        model_type='image_to_image',
        model_uri='wdnet_v0.1',
        model_device=args.device,

        log_level=args.log_level,
    )
    model_server.run_forever()
