// Matrix
// -------------------------------------------------
float4x4 matW : World;
float4x4 matV : View;
float4x4 matVI : ViewInverse;
float4x4 matWV : WorldView;
float4x4 matWVP : WorldViewProjection;
float4x4 matBones[58];

// Material
// -------------------------------------------------
float3 diffuseColor;
float3 specularColor;
float specularPower;

// Light
// -------------------------------------------------
float3 ambientLightColor;
float3 light1Position;
float3 light1Color;
float3 light2Position;
float3 light2Color;

// Textures
// -------------------------------------------------
float2 uv0Tile;
texture diffuseTexture1 : Diffuse;

sampler2D diffuse1Sampler = sampler_state {
    texture = <diffuseTexture1>;
    MagFilter = Linear;
    MinFilter = Linear;
    MipFilter = Linear;
};

// Application to Vertex
// -------------------------------------------------
struct a2v
{
    float4 position		: POSITION;
    float3 normal		: NORMAL;
    float2 uv0			: TEXCOORD0;
    float4 boneIndex	: BLENDINDICES0;
    float4 boneWeight	: BLENDWEIGHT0;
};

// Vertex to Fragment
// -------------------------------------------------
struct v2f
{
    float4 hposition	: POSITION;
    float2 uv0			: TEXCOORD0;
    float3 normal		: TEXCOORD1;
    float3 lightVec1	: TEXCOORD2;
    float3 lightVec2	: TEXCOORD3;
    float3 eyeVec		: TEXCOORD4;
};

v2f animatedModelVS(a2v IN)
{
	v2f OUT;
	
	// Calculate the final bone transformation matrix
	float4x4 matTransform = matBones[IN.boneIndex.x] * IN.boneWeight.x;
	matTransform += matBones[IN.boneIndex.y] * IN.boneWeight.y;
	matTransform += matBones[IN.boneIndex.z] * IN.boneWeight.z;
	float finalWeight = 1.0f - (IN.boneWeight.x + IN.boneWeight.y + IN.boneWeight.z);
	matTransform += matBones[IN.boneIndex.w] * finalWeight;
	
	// Transform vertex and normal
	float4 position = mul(IN.position, matTransform);
	float3 normal = mul(IN.normal, matTransform);
	OUT.hposition = mul(position, matWVP);
	OUT.normal = mul(normal, matWV);
	
	// Calculate light and eye vectors
	float4 worldPosition = mul(position, matW);
	OUT.eyeVec = mul(matVI[3].xyz - worldPosition, matV);
    OUT.lightVec1 = mul(light1Position - worldPosition, matV);
    OUT.lightVec2 = mul(light2Position - worldPosition, matV);
	OUT.uv0 = IN.uv0;

    return OUT;
}

void phongShading(in float3 normal, in float3 lightVec, in float3 halfwayVec,
	in float3 lightColor, out float3 diffuseColor, out float3 specularColor)
{
	float diffuseInt = saturate(dot(normal, lightVec));
	diffuseColor = diffuseInt * lightColor;
	
	float specularInt = saturate(dot(normal, halfwayVec));
	specularInt = pow(specularInt, specularPower);
	specularColor = specularInt * lightColor;
}

float4 animatedModelPS(v2f IN): COLOR0
{
    // Normalize all input vectors
	float3 normal = normalize(IN.normal);
    float3 eyeVec = normalize(IN.eyeVec);
    float3 lightVec1 = normalize(IN.lightVec1);
    float3 lightVec2 = normalize(IN.lightVec2);
        
    // Calculate halfway vectors
    float3 halfwayVec1 = normalize(lightVec1 + eyeVec);
    float3 halfwayVec2 = normalize(lightVec2 + eyeVec);

	// Calculate diffuse and specular color for each light        
	float3 diffuseColor1, diffuseColor2;
	float3 specularColor1, specularColor2;
    phongShading(normal, lightVec1, halfwayVec1, light1Color, diffuseColor1, specularColor1);
    phongShading(normal, lightVec2, halfwayVec2, light2Color, diffuseColor2, specularColor2);
    
    // Read texture diffuse color
    float4 materialColor = tex2D(diffuse1Sampler, IN.uv0);

    // Phong lighting result    
    float4 finalColor;
    finalColor.a = 1.0f;
    finalColor.rgb = materialColor * 
		( (diffuseColor1 + diffuseColor2) * diffuseColor + ambientLightColor) + 
		(specularColor1 + specularColor2) * specularColor ;
    
    return finalColor;
}

technique AnimatedModel
{
    pass p0
    {
		ZWriteEnable = true;
        VertexShader = compile vs_2_0 animatedModelVS();
		PixelShader = compile ps_2_0 animatedModelPS();
    }
}
