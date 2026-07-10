import * as THREE from "three";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const activeViewers = new Set();

function fitObjectToView(object, camera, controls, targetSize = 0.55) {
  object.updateMatrixWorld(true);

  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  object.position.sub(center);
  object.updateMatrixWorld(true);

  const maxDim = Math.max(size.x, size.y, size.z, 0.001);
  object.scale.setScalar(targetSize / maxDim);
  object.updateMatrixWorld(true);

  const fittedBox = new THREE.Box3().setFromObject(object);
  const fittedCenter = fittedBox.getCenter(new THREE.Vector3());
  object.position.sub(fittedCenter);
  object.updateMatrixWorld(true);

  camera.position.set(0, 0, 10);
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();

  if (controls) {
    controls.target.set(0, 0, 0);
    controls.update();
  }
}

function createScene(canvas) {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 0, 10);

  const ambient = new THREE.AmbientLight(0xffffff, 0.75);
  const key = new THREE.DirectionalLight(0xffffff, 1.05);
  key.position.set(2.5, 4, 3.5);
  const fill = new THREE.DirectionalLight(0xdce7ff, 0.55);
  fill.position.set(-3, 1.5, -2);
  scene.add(ambient, key, fill);

  return { renderer, scene, camera };
}

async function loadModel(url, source) {
  if (source === "glb" || url.toLowerCase().endsWith(".glb") || url.toLowerCase().endsWith(".gltf")) {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(url);
    return gltf.scene;
  }

  const loader = new STLLoader();
  const geometry = await loader.loadAsync(url);
  const material = new THREE.MeshStandardMaterial({
    color: 0xb8c2d1,
    metalness: 0.15,
    roughness: 0.55,
    flatShading: false
  });
  return new THREE.Mesh(geometry, material);
}

export async function createModelViewer(canvas, { url, source, label = "", onInteract = null }) {
  const { renderer, scene, camera } = createScene(canvas);
  const root = await loadModel(url, source);
  scene.add(root);

  const controls = new OrbitControls(camera, canvas);
  controls.enablePan = true;
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.75;
  controls.zoomSpeed = 5.5;
  controls.minDistance = 2.5;
  controls.maxDistance = 10;
  controls.target.set(0, 0, 0);

  let dragging = false;
  let moved = false;
  let hasInteracted = false;
  const markInteracted = () => {
    if (hasInteracted) return;
    hasInteracted = true;
    onInteract?.(canvas);
  };

  controls.addEventListener("start", () => {
    dragging = true;
    moved = false;
  });
  controls.addEventListener("change", () => {
    if (dragging) moved = true;
  });
  controls.addEventListener("end", () => {
    if (moved) markInteracted();
    dragging = false;
  });

  const resize = () => {
    const stage = canvas.parentElement;
    const width = Math.max(stage?.clientWidth || canvas.clientWidth || 236, 120);
    const height = Math.max(stage?.clientHeight || canvas.clientHeight || 236, 120);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    fitObjectToView(root, camera, controls);
  };

  fitObjectToView(root, camera, controls);
  controls.update();

  let frameId = null;
  const renderLoop = () => {
    controls.update();
    renderer.render(scene, camera);
    frameId = window.requestAnimationFrame(renderLoop);
  };

  await new Promise((resolve) => window.requestAnimationFrame(resolve));
  resize();
  renderLoop();
  await new Promise((resolve) => window.requestAnimationFrame(resolve));
  resize();

  const viewer = {
    canvas,
    label,
    url,
    source,
    controls,
    resize,
    dispose() {
      if (frameId !== null) {
        window.cancelAnimationFrame(frameId);
        frameId = null;
      }
      controls.dispose();
      root.traverse((child) => {
        if (child.isMesh) {
          child.geometry?.dispose?.();
          if (Array.isArray(child.material)) {
            child.material.forEach((mat) => mat.dispose?.());
          } else {
            child.material?.dispose?.();
          }
        }
      });
      renderer.dispose();
      activeViewers.delete(viewer);
    }
  };

  activeViewers.add(viewer);
  return viewer;
}

export function disposeAllModelViewers() {
  for (const viewer of [...activeViewers]) {
    viewer.dispose();
  }
}

export function resizeAllModelViewers() {
  for (const viewer of activeViewers) {
    viewer.resize();
  }
}
