/**
 * GhostLayer design cue: spatial navigation in a monochrome orbital field.
 * The globe is intentionally functional: teal marks verified paths, rust marks constrained paths,
 * and each marker shares a single focus model with the site hot bar and information panels.
 */
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, Line, OrbitControls, Stars } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

export type DestinationId = "info" | "control" | "memory" | "contact";

export type Destination = {
  id: DestinationId;
  label: string;
  shortLabel: string;
  latitude: number;
  longitude: number;
  tone: "signal" | "risk" | "neutral";
};

export const DESTINATIONS: Destination[] = [
  { id: "info", label: "Info", shortLabel: "Information field", latitude: 26, longitude: -12, tone: "neutral" },
  { id: "control", label: "Decision Lab", shortLabel: "Control loop", latitude: -11, longitude: 67, tone: "signal" },
  { id: "memory", label: "Memory", shortLabel: "Infrastructure memory", latitude: 32, longitude: 137, tone: "signal" },
  { id: "contact", label: "Contact us", shortLabel: "Contact window", latitude: -27, longitude: -84, tone: "risk" },
];

const byId = Object.fromEntries(DESTINATIONS.map((destination) => [destination.id, destination])) as Record<DestinationId, Destination>;

function globePosition(latitude: number, longitude: number, radius = 2.06) {
  const phi = THREE.MathUtils.degToRad(latitude);
  const theta = THREE.MathUtils.degToRad(longitude);
  return new THREE.Vector3(
    radius * Math.cos(phi) * Math.sin(theta),
    radius * Math.sin(phi),
    radius * Math.cos(phi) * Math.cos(theta),
  );
}

function Marker({ destination, active, onSelect }: { destination: Destination; active: boolean; onSelect: (id: DestinationId) => void }) {
  const point = useMemo(() => globePosition(destination.latitude, destination.longitude), [destination]);
  const tone = destination.tone === "risk" ? "#f16d52" : destination.tone === "signal" ? "#42d8bb" : "#edf1ed";
  const halo = destination.tone === "risk" ? "#f16d52" : "#42d8bb";

  return (
    <group position={point}>
      <mesh
        onClick={(event) => {
          event.stopPropagation();
          onSelect(destination.id);
        }}
        onPointerOver={(event) => {
          event.stopPropagation();
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          document.body.style.cursor = "auto";
        }}
      >
        <sphereGeometry args={[active ? 0.105 : 0.075, 24, 24]} />
        <meshBasicMaterial color={tone} />
      </mesh>
      <mesh scale={active ? 1.8 : 1.25}>
        <sphereGeometry args={[0.11, 24, 24]} />
        <meshBasicMaterial color={halo} transparent opacity={active ? 0.24 : 0.11} />
      </mesh>
      <Html center distanceFactor={9} style={{ pointerEvents: "none" }}>
        <div className={`globe-marker-label ${active ? "is-active" : ""} globe-marker-label--${destination.tone}`}>
          <span className="marker-tick" />
          {destination.label}
        </div>
      </Html>
    </group>
  );
}

function NetworkLines({ activeDestination }: { activeDestination: DestinationId | null }) {
  const paths = useMemo(() => {
    const pairs: [DestinationId, DestinationId][] = [
      ["info", "control"],
      ["control", "memory"],
      ["memory", "contact"],
      ["contact", "info"],
      ["info", "memory"],
    ];
    return pairs.map(([a, b]) => {
      const pointA = globePosition(byId[a].latitude, byId[a].longitude, 2.08);
      const pointB = globePosition(byId[b].latitude, byId[b].longitude, 2.08);
      const midpoint = pointA.clone().add(pointB).multiplyScalar(0.5).normalize().multiplyScalar(2.7);
      return { id: `${a}-${b}`, points: new THREE.QuadraticBezierCurve3(pointA, midpoint, pointB).getPoints(40), includesActive: activeDestination === a || activeDestination === b };
    });
  }, [activeDestination]);

  return (
    <group>
      {paths.map((path) => (
        <Line
          key={path.id}
          points={path.points}
          color={path.includesActive ? "#42d8bb" : "#6c7778"}
          lineWidth={path.includesActive ? 1.15 : 0.45}
          transparent
          opacity={path.includesActive ? 0.72 : 0.29}
        />
      ))}
    </group>
  );
}

function DecisionTrace({ activeDestination }: { activeDestination: DestinationId | null }) {
  const trace = useMemo(() => {
    const detect = globePosition(byId.info.latitude, byId.info.longitude, 2.1);
    const verify = globePosition(byId.control.latitude, byId.control.longitude, 2.1);
    const record = globePosition(byId.memory.latitude, byId.memory.longitude, 2.1);
    const firstControl = detect.clone().add(verify).multiplyScalar(0.5).normalize().multiplyScalar(2.92);
    const secondControl = verify.clone().add(record).multiplyScalar(0.5).normalize().multiplyScalar(2.92);
    return [
      ...new THREE.QuadraticBezierCurve3(detect, firstControl, verify).getPoints(26),
      ...new THREE.QuadraticBezierCurve3(verify, secondControl, record).getPoints(26),
    ];
  }, []);

  return (
    <group>
      <Line points={trace} color="#42d8bb" lineWidth={activeDestination ? 1.65 : 0.92} transparent opacity={activeDestination ? 0.95 : 0.61} />
      {[byId.info, byId.control, byId.memory].map((destination, index) => (
        <mesh key={destination.id} position={globePosition(destination.latitude, destination.longitude, 2.12)}>
          <ringGeometry args={[0.075 + index * 0.008, 0.101 + index * 0.008, 24]} />
          <meshBasicMaterial color="#42d8bb" transparent opacity={0.85} side={THREE.DoubleSide} />
        </mesh>
      ))}
    </group>
  );
}

function createGhostSurfaceTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 512;
  const context = canvas.getContext("2d");
  if (!context) return new THREE.Texture();

  context.fillStyle = "#0b1013";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const seeded = (value: number) => {
    const sample = Math.sin(value * 12.9898) * 43758.5453;
    return sample - Math.floor(sample);
  };

  for (let index = 0; index < 180; index += 1) {
    const x = seeded(index + 1) * canvas.width;
    const y = seeded(index + 8) * canvas.height;
    const radius = 10 + seeded(index + 15) * 90;
    const alpha = 0.015 + seeded(index + 23) * 0.075;
    const gradient = context.createRadialGradient(x, y, 0, x, y, radius);
    gradient.addColorStop(0, `rgba(194, 231, 220, ${alpha})`);
    gradient.addColorStop(0.46, `rgba(61, 124, 113, ${alpha * 0.46})`);
    gradient.addColorStop(1, "rgba(7, 11, 13, 0)");
    context.fillStyle = gradient;
    context.fillRect(x - radius, y - radius, radius * 2, radius * 2);
  }

  context.globalCompositeOperation = "screen";
  for (let index = 0; index < 54; index += 1) {
    const x = seeded(index + 63) * canvas.width;
    const y = seeded(index + 91) * canvas.height;
    context.fillStyle = `rgba(146, 186, 174, ${0.028 + seeded(index + 119) * 0.055})`;
    context.fillRect(x, y, 2 + seeded(index) * 14, 0.8 + seeded(index + 27) * 2.2);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  return texture;
}

function EtherealHalo() {
  const motes = useMemo(() => {
    const positions = new Float32Array(640 * 3);
    for (let index = 0; index < 640; index += 1) {
      const theta = (index * 2.399963229728653) % (Math.PI * 2);
      const phi = Math.acos(1 - (2 * (index + 0.5)) / 640);
      const radius = 2.18 + ((index * 29) % 23) / 100;
      positions[index * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[index * 3 + 1] = radius * Math.cos(phi);
      positions[index * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);
    }
    return positions;
  }, []);

  return (
    <group>
      <mesh scale={1.11}>
        <sphereGeometry args={[2, 72, 72]} />
        <meshBasicMaterial color="#79d2c1" transparent opacity={0.025} side={THREE.BackSide} depthWrite={false} />
      </mesh>
      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[motes, 3]} />
        </bufferGeometry>
        <pointsMaterial color="#c9f5eb" size={0.016} transparent opacity={0.48} sizeAttenuation depthWrite={false} />
      </points>
    </group>
  );
}

function GlobeBody({ activeDestination, onSelect }: { activeDestination: DestinationId | null; onSelect: (id: DestinationId) => void }) {
  const group = useRef<THREE.Group | null>(null);
  const surfaceTexture = useMemo(() => createGhostSurfaceTexture(), []);
  const sparkPoints = useMemo(() => {
    const points: { position: THREE.Vector3; color: string }[] = [];
    for (let index = 0; index < 88; index += 1) {
      const latitude = ((index * 47) % 154) - 77;
      const longitude = ((index * 73) % 360) - 180;
      const red = index % 17 === 0;
      const green = index % 7 === 0;
      points.push({ position: globePosition(latitude, longitude, 2.015), color: red ? "#f16d52" : green ? "#42d8bb" : "#8a9495" });
    }
    return points;
  }, []);

  useFrame((state) => {
    if (!group.current) return;
    const drift = activeDestination ? 0.0008 : 0.002;
    group.current.rotation.y += drift;
    group.current.rotation.x = THREE.MathUtils.lerp(group.current.rotation.x, Math.sin(state.clock.elapsedTime * 0.17) * 0.025, 0.025);
  });

  return (
    <group ref={group}>
      <mesh>
        <sphereGeometry args={[2, 72, 72]} />
        <meshStandardMaterial map={surfaceTexture} color="#536662" metalness={0.42} roughness={0.72} emissive="#081113" emissiveIntensity={0.45} />
      </mesh>
      <mesh scale={1.014}>
        <sphereGeometry args={[2, 72, 72]} />
        <meshBasicMaterial color="#7cccbf" transparent opacity={0.07} depthWrite={false} />
      </mesh>
      <mesh scale={1.003}>
        <sphereGeometry args={[2, 40, 40]} />
        <meshBasicMaterial color="#d9e3df" wireframe transparent opacity={0.17} />
      </mesh>
      <mesh scale={1.034}>
        <sphereGeometry args={[2, 36, 20]} />
        <meshBasicMaterial color="#67d6c4" wireframe transparent opacity={0.06} />
      </mesh>
      {sparkPoints.map((point, index) => (
        <mesh key={index} position={point.position}>
          <sphereGeometry args={[0.018, 8, 8]} />
          <meshBasicMaterial color={point.color} transparent opacity={0.8} />
        </mesh>
      ))}
      <NetworkLines activeDestination={activeDestination} />
      <DecisionTrace activeDestination={activeDestination} />
      <EtherealHalo />
      {DESTINATIONS.map((destination) => <Marker key={destination.id} destination={destination} active={activeDestination === destination.id} onSelect={onSelect} />)}
    </group>
  );
}

function CameraDirector({ activeDestination, navigationStep }: { activeDestination: DestinationId | null; navigationStep: number }) {
  const controls = useRef<any>(null);
  const { camera } = useThree();
  const shouldAnimate = useRef(true);

  useEffect(() => {
    shouldAnimate.current = true;
  }, [activeDestination, navigationStep]);

  useFrame((_, delta) => {
    if (!shouldAnimate.current) return;
    const focusPoint = activeDestination ? globePosition(byId[activeDestination].latitude, byId[activeDestination].longitude, 1.9) : new THREE.Vector3(0, 0, 0);
    const direction = activeDestination ? globePosition(byId[activeDestination].latitude, byId[activeDestination].longitude, 1).normalize() : new THREE.Vector3(0, 0, 1);
    const cameraPosition = direction.multiplyScalar(activeDestination ? 4.05 : 6.2);
    const interpolation = 1 - Math.pow(0.001, delta);
    camera.position.lerp(cameraPosition, interpolation);
    controls.current?.target.lerp(activeDestination ? focusPoint.multiplyScalar(0.22) : new THREE.Vector3(0, 0, 0), interpolation);
    controls.current?.update();
    if (camera.position.distanceTo(cameraPosition) < 0.018) shouldAnimate.current = false;
  });

  return <OrbitControls ref={controls} enablePan={false} minDistance={3.1} maxDistance={8.4} rotateSpeed={0.58} zoomSpeed={0.55} dampingFactor={0.07} enableDamping />;
}

export default function GlobeNavigator({ activeDestination, navigationStep, onSelect }: { activeDestination: DestinationId | null; navigationStep: number; onSelect: (id: DestinationId) => void }) {
  return (
    <div className="globe-navigator" aria-label="Interactive 3D site navigator">
      <Canvas camera={{ position: [0, 0, 6.2], fov: 43 }} dpr={[1, 2]} gl={{ antialias: true, alpha: true }}>
        <ambientLight intensity={0.52} />
        <directionalLight position={[5, 4, 6]} intensity={2.2} color="#e8f1ec" />
        <pointLight position={[-4, -1, 3]} intensity={3.4} color="#2bc9ad" distance={8} />
        <pointLight position={[2, -3, -4]} intensity={1.4} color="#f16d52" distance={7} />
        <Stars radius={58} depth={45} count={1120} factor={3.1} saturation={0} fade speed={0.18} />
        <group rotation={[0.25, -0.22, 0]}>
          <GlobeBody activeDestination={activeDestination} onSelect={onSelect} />
        </group>
        <mesh rotation={[Math.PI / 2.45, 0.2, 0.55]}>
          <torusGeometry args={[2.45, 0.009, 8, 180]} />
          <meshBasicMaterial color="#eaf0ec" transparent opacity={0.21} />
        </mesh>
        <mesh rotation={[Math.PI / 1.92, -0.68, -0.3]}>
          <torusGeometry args={[2.78, 0.004, 8, 180]} />
          <meshBasicMaterial color="#42d8bb" transparent opacity={0.24} />
        </mesh>
        <CameraDirector activeDestination={activeDestination} navigationStep={navigationStep} />
      </Canvas>
      <div className="globe-navigator__hint"><span className="hint-icon">⊹</span> Drag the field. Select a marked location.</div>
    </div>
  );
}
