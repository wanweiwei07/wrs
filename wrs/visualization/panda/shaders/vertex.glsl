#version 150 core

layout (location = 0) in vec3 aPos;   // Position attribute
layout (location = 1) in vec3 aNormal; // Normal attribute

out vec3 Normal;  // For passing to the fragment shader
out vec3 FragPos; // Fragment position

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;

    gl_Position = projection * view * vec4(FragPos, 1.0);
}