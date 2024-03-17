#version 150 core

out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec4 lightColor;
uniform vec4 objectColor;

void main()
{
    // Ambient
    float ambientStrength = 0.1;
    vec4 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec4 diffuse = diff * lightColor;

    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec4 specular = specularStrength * spec * lightColor;

    // Combine results
    vec4 result = (ambient + diffuse + specular) * objectColor;

    // Quantize the result to achieve the cel-shaded look
    float quantizeFactor = 0.2; // Adjust for more or fewer tones
    result = floor(result / quantizeFactor + 0.5) * quantizeFactor;

    FragColor = result;
}