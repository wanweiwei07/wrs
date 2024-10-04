// Set up the scene, camera, and renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const shaderMaterial = new THREE.ShaderMaterial({
    vertexShader: `
        varying vec3 vNormal;
        void main() {
            vNormal = normalize(normalMatrix * normal);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        varying vec3 vNormal;
        void main() {
            vec3 light = normalize(vec3(1.0, 1.0, 1.0)); // Light direction
            float shading = dot(vNormal, light) * 0.5 + 0.5;
            vec3 color = vec3(0.0, 0.0, 1.0); // Base color
            float bands = 3.0; // Number of color bands
            float bandIndex = floor(shading * bands) / bands;
            gl_FragColor = vec4(color * bandIndex, 1.0);
        }
    `
});

// Create a cube
const geometry = new THREE.BoxGeometry(); // 1x1x1 cube
const cube = new THREE.Mesh(geometry, shaderMaterial);
scene.add(cube);

// Position the camera
camera.position.z = 5;

// Lighting (optional for MeshBasicMaterial, but added for future customization)
const light = new THREE.PointLight(0xffffff, 1, 500);
light.position.set(10, 10, 10);
scene.add(light);

// Animation function to rotate the cube
function animate() {
    requestAnimationFrame(animate);

    // Rotate the cube
    cube.rotation.x += 0.01;
    cube.rotation.y += 0.01;

    // Render the scene from the perspective of the camera
    renderer.render(scene, camera);
}

// Start the animation loop
animate();

controls = new THREE.OrbitControls(camera, renderer.domElement);