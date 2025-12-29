<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import * as THREE from 'three'
import { STLLoader } from 'three/addons/loaders/STLLoader.js'
import { STLExporter } from 'three/addons/exporters/STLExporter.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { Brush, Evaluator, SUBTRACTION } from 'three-bvh-csg'
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js'

// A face is a group of triangles that share the same normal (within tolerance)
interface Face {
  normal: THREE.Vector3
  triangleIndices: number[]
  center: THREE.Vector3
}

const containerRef = ref<HTMLDivElement | null>(null)
const fileInputRef = ref<HTMLInputElement | null>(null)
const fileName = ref<string | null>(null)
const isLoading = ref(false)
const errorMessage = ref<string | null>(null)
// Selected faces for wall definition (up to 2 faces)
const selectedFaces = ref<number[]>([])
const faces = ref<Face[]>([])
const isDrilling = ref(false)

// Can drill when exactly 2 faces are selected and wall data is computed
const canDrill = computed(() => selectedFaces.value.length === 2 && currentWallData !== null)

let scene: THREE.Scene
let camera: THREE.PerspectiveCamera
let renderer: THREE.WebGLRenderer
let controls: OrbitControls
let currentMesh: THREE.Mesh | null = null
let animationFrameId: number
let raycaster: THREE.Raycaster
let mouse: THREE.Vector2
let mouseDownPosition: { x: number; y: number } | null = null

// Maps triangle index to face index for quick lookup
let triangleToFaceMap: Map<number, number> = new Map()

const BASE_COLOR = new THREE.Color(0x00d4aa)
const HIGHLIGHT_COLOR = new THREE.Color(0xff6b6b)
const OVERLAP_COLOR = new THREE.Color(0x9966ff) // Purple for overlap highlight
const DIM_SELECTED_COLOR = new THREE.Color(0x666699) // Dim color to show full selected faces

// Store wall parameters for drilling and visualization
interface WallData {
  // 2D intersection polygon in local coordinates
  intersection: THREE.Vector2[]
  // Plane basis for converting 2D back to 3D
  planeOrigin: THREE.Vector3
  uAxis: THREE.Vector3
  vAxis: THREE.Vector3
  // Drill axis (direction to drill through)
  drillAxis: THREE.Vector3
  // Start and end positions along drill axis
  drillStart: number
  drillEnd: number
  // 3D vertices of intersection polygon on each face (for overlay visualization)
  face1Vertices: THREE.Vector3[]
  face2Vertices: THREE.Vector3[]
  // Face normals (for positioning overlays)
  face1Normal: THREE.Vector3
  face2Normal: THREE.Vector3
}
let currentWallData: WallData | null = null

// Overlay meshes showing the exact intersection on each face
let overlayMeshes: THREE.Mesh[] = []

// Create a canonical edge key from two vertex positions
const makeEdgeKey = (v1: THREE.Vector3, v2: THREE.Vector3): string => {
  // Round to avoid floating point issues, then sort to ensure consistent key
  const precision = 1e6
  const round = (n: number) => Math.round(n * precision)

  const p1 = `${round(v1.x)},${round(v1.y)},${round(v1.z)}`
  const p2 = `${round(v2.x)},${round(v2.y)},${round(v2.z)}`

  return p1 < p2 ? `${p1}|${p2}` : `${p2}|${p1}`
}

// Get the normal for a triangle
const getTriangleNormal = (
  normalAttr: THREE.BufferAttribute,
  triangleIndex: number,
): THREE.Vector3 => {
  const i = triangleIndex * 3
  return new THREE.Vector3(normalAttr.getX(i), normalAttr.getY(i), normalAttr.getZ(i)).normalize()
}

// Get vertex position from geometry
const getVertex = (positionAttr: THREE.BufferAttribute, vertexIndex: number): THREE.Vector3 => {
  return new THREE.Vector3(
    positionAttr.getX(vertexIndex),
    positionAttr.getY(vertexIndex),
    positionAttr.getZ(vertexIndex),
  )
}

// Project a 3D point onto a plane, returning 2D coordinates in the plane's local frame
const projectToPlane = (
  point: THREE.Vector3,
  planeOrigin: THREE.Vector3,
  planeNormal: THREE.Vector3,
  uAxis: THREE.Vector3,
  vAxis: THREE.Vector3,
): THREE.Vector2 => {
  const relative = point.clone().sub(planeOrigin)
  // Project onto the plane by removing the normal component
  const distToPlane = relative.dot(planeNormal)
  const projected = relative.clone().sub(planeNormal.clone().multiplyScalar(distToPlane))
  return new THREE.Vector2(projected.dot(uAxis), projected.dot(vAxis))
}

// Compute 2D convex hull using Graham scan algorithm
const convexHull2D = (points: THREE.Vector2[]): THREE.Vector2[] => {
  if (points.length < 3) return points.slice()

  // Find the point with lowest y (and lowest x if tied)
  let start = 0
  for (let i = 1; i < points.length; i++) {
    const p = points[i]!
    const s = points[start]!
    if (p.y < s.y || (p.y === s.y && p.x < s.x)) {
      start = i
    }
  }

  const startPoint = points[start]!

  // Sort points by polar angle from start point
  const sorted = points
    .filter((_, i) => i !== start)
    .map((p) => ({
      point: p,
      angle: Math.atan2(p.y - startPoint.y, p.x - startPoint.x),
      dist: p.distanceTo(startPoint),
    }))
    .sort((a, b) => a.angle - b.angle || a.dist - b.dist)
    .map((p) => p.point)

  // Cross product to determine turn direction
  const cross = (o: THREE.Vector2, a: THREE.Vector2, b: THREE.Vector2): number =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

  const hull: THREE.Vector2[] = [startPoint]

  for (const p of sorted) {
    while (hull.length >= 2 && cross(hull[hull.length - 2]!, hull[hull.length - 1]!, p) <= 0) {
      hull.pop()
    }
    hull.push(p)
  }

  return hull
}

// Sutherland-Hodgman polygon clipping algorithm for convex polygons
const clipPolygon = (subject: THREE.Vector2[], clip: THREE.Vector2[]): THREE.Vector2[] => {
  if (subject.length < 3 || clip.length < 3) return []

  let output = subject.slice()

  for (let i = 0; i < clip.length; i++) {
    if (output.length === 0) return []

    const input = output
    output = []

    const edgeStart = clip[i]!
    const edgeEnd = clip[(i + 1) % clip.length]!

    const inside = (p: THREE.Vector2): boolean => {
      return (
        (edgeEnd.x - edgeStart.x) * (p.y - edgeStart.y) -
          (edgeEnd.y - edgeStart.y) * (p.x - edgeStart.x) >=
        0
      )
    }

    const intersect = (p1: THREE.Vector2, p2: THREE.Vector2): THREE.Vector2 => {
      const dc = new THREE.Vector2(edgeStart.x - edgeEnd.x, edgeStart.y - edgeEnd.y)
      const dp = new THREE.Vector2(p1.x - p2.x, p1.y - p2.y)
      const n1 = (edgeStart.x - p1.x) * dc.y - (edgeStart.y - p1.y) * dc.x
      const n2 = dp.x * dc.y - dp.y * dc.x
      if (Math.abs(n2) < 1e-10) return p1.clone()
      const t = n1 / n2
      return new THREE.Vector2(p1.x + t * dp.x, p1.y + t * dp.y)
    }

    for (let j = 0; j < input.length; j++) {
      const current = input[j]!
      const next = input[(j + 1) % input.length]!
      const currentInside = inside(current)
      const nextInside = inside(next)

      if (currentInside) {
        output.push(current)
        if (!nextInside) {
          output.push(intersect(current, next))
        }
      } else if (nextInside) {
        output.push(intersect(current, next))
      }
    }
  }

  return output
}

// Create orthonormal basis vectors for a plane given its normal
const createPlaneBasis = (normal: THREE.Vector3): { u: THREE.Vector3; v: THREE.Vector3 } => {
  // Choose a vector not parallel to normal
  const up = Math.abs(normal.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0)
  const u = new THREE.Vector3().crossVectors(up, normal).normalize()
  const v = new THREE.Vector3().crossVectors(normal, u).normalize()
  return { u, v }
}

// Calculate the signed distance of a face's plane from origin along a reference normal
const getFacePlaneDistance = (
  face: Face,
  positionAttr: THREE.BufferAttribute,
  referenceNormal: THREE.Vector3,
): number => {
  // Use the first vertex of the face to calculate plane distance
  const firstTriIndex = face.triangleIndices[0]!
  const vertex = getVertex(positionAttr, firstTriIndex * 3)
  // Use reference normal for consistent distance measurement
  return vertex.dot(referenceNormal)
}

interface WallResult {
  geometry: THREE.BufferGeometry
  wallData: WallData
}

// Check if two faces are coplanar (same plane, not just parallel)
const areFacesCoplanar = (
  face1: Face,
  face2: Face,
  positionAttr: THREE.BufferAttribute,
): boolean => {
  const normalTolerance = 0.01
  const distanceTolerance = 0.1

  // Check if normals are the same or opposite (parallel planes)
  const normalDot = face1.normal.dot(face2.normal)
  const isParallel = Math.abs(Math.abs(normalDot) - 1) < normalTolerance

  if (!isParallel) return false

  // Check if on the same plane (same distance from origin)
  const refNormal = face1.normal
  const refDistance = getFacePlaneDistance(face1, positionAttr, refNormal)

  const useNormal = normalDot < 0 ? refNormal.clone().negate() : refNormal
  const faceDistance = getFacePlaneDistance(face2, positionAttr, useNormal)
  const adjustedRefDistance = normalDot < 0 ? -refDistance : refDistance

  return Math.abs(faceDistance - adjustedRefDistance) < distanceTolerance
}

// Check if two faces share an edge (have two vertices in common)
const facesShareEdge = (face1: Face, face2: Face, positionAttr: THREE.BufferAttribute): boolean => {
  const precision = 1e6
  const round = (n: number) => Math.round(n * precision)

  // Collect all vertex keys from face1
  const face1Vertices = new Set<string>()
  for (const triIndex of face1.triangleIndices) {
    const baseVertex = triIndex * 3
    for (let v = 0; v < 3; v++) {
      const vertex = getVertex(positionAttr, baseVertex + v)
      const key = `${round(vertex.x)},${round(vertex.y)},${round(vertex.z)}`
      face1Vertices.add(key)
    }
  }

  // Count how many vertices from face2 are shared with face1
  let sharedCount = 0
  const checked = new Set<string>()
  for (const triIndex of face2.triangleIndices) {
    const baseVertex = triIndex * 3
    for (let v = 0; v < 3; v++) {
      const vertex = getVertex(positionAttr, baseVertex + v)
      const key = `${round(vertex.x)},${round(vertex.y)},${round(vertex.z)}`
      if (!checked.has(key)) {
        checked.add(key)
        if (face1Vertices.has(key)) {
          sharedCount++
          if (sharedCount >= 2) return true // Two shared vertices = shared edge
        }
      }
    }
  }

  return false
}

// Find all faces that are coplanar AND connected (share edges) with the given face
// Uses flood-fill to only include directly connected coplanar faces
const findConnectedCoplanarFaces = (
  faceIndex: number,
  allFaces: Face[],
  positionAttr: THREE.BufferAttribute,
): number[] => {
  const referenceFace = allFaces[faceIndex]
  if (!referenceFace) return [faceIndex]

  const visited = new Set<number>()
  const result: number[] = []
  const queue: number[] = [faceIndex]

  while (queue.length > 0) {
    const currentIndex = queue.pop()!
    if (visited.has(currentIndex)) continue
    visited.add(currentIndex)

    const currentFace = allFaces[currentIndex]
    if (!currentFace) continue

    // Check if coplanar with the reference face
    if (currentIndex !== faceIndex && !areFacesCoplanar(referenceFace, currentFace, positionAttr)) {
      continue
    }

    result.push(currentIndex)

    // Find neighboring faces (share an edge) and add to queue
    for (let i = 0; i < allFaces.length; i++) {
      if (visited.has(i)) continue
      const neighborFace = allFaces[i]!

      if (facesShareEdge(currentFace, neighborFace, positionAttr)) {
        queue.push(i)
      }
    }
  }

  return result.length > 0 ? result : [faceIndex]
}

// Get combined vertices from multiple faces
const getCombinedFaceVertices = (
  faceIndices: number[],
  allFaces: Face[],
  positionAttr: THREE.BufferAttribute,
): THREE.Vector3[] => {
  const precision = 1e6
  const round = (n: number) => Math.round(n * precision)
  const seen = new Set<string>()
  const vertices: THREE.Vector3[] = []

  for (const faceIndex of faceIndices) {
    const face = allFaces[faceIndex]
    if (!face) continue

    for (const triIndex of face.triangleIndices) {
      const baseVertex = triIndex * 3
      for (let v = 0; v < 3; v++) {
        const vertex = getVertex(positionAttr, baseVertex + v)
        const key = `${round(vertex.x)},${round(vertex.y)},${round(vertex.z)}`
        if (!seen.has(key)) {
          seen.add(key)
          vertices.push(vertex)
        }
      }
    }
  }

  return vertices
}

// Get combined center from multiple faces
const getCombinedFaceCenter = (faceIndices: number[], allFaces: Face[]): THREE.Vector3 => {
  const center = new THREE.Vector3()
  let count = 0

  for (const faceIndex of faceIndices) {
    const face = allFaces[faceIndex]
    if (!face) continue
    center.add(face.center)
    count++
  }

  if (count > 0) {
    center.divideScalar(count)
  }

  return center
}

// Create wall geometry from two selected faces (considering all coplanar faces)
const createWallGeometry = (
  face1Indices: number[],
  face2Indices: number[],
  allFaces: Face[],
  positionAttr: THREE.BufferAttribute,
): WallResult | null => {
  const normalTolerance = 0.01

  // Use the first face of each group as reference for normals
  const face1 = allFaces[face1Indices[0]!]
  const face2 = allFaces[face2Indices[0]!]

  if (!face1 || !face2) return null

  // Check if normals are the same (parallel planes) or opposite
  const normalDot = face1.normal.dot(face2.normal)
  const sameNormal = Math.abs(normalDot) > 1 - normalTolerance

  if (sameNormal) {
    return createSameNormalWall(face1Indices, face2Indices, allFaces, positionAttr, normalDot)
  } else {
    return createDifferentNormalWall(face1Indices, face2Indices, allFaces, positionAttr)
  }
}

// Create wall for faces with same/opposite normals (cuboid)
const createSameNormalWall = (
  face1Indices: number[],
  face2Indices: number[],
  allFaces: Face[],
  positionAttr: THREE.BufferAttribute,
  normalDot: number,
): WallResult | null => {
  // Use the first face of each group as reference for normals
  const face1 = allFaces[face1Indices[0]!]!
  const face2 = allFaces[face2Indices[0]!]!

  // Use face1's normal as reference (flip face2's if opposite)
  const normal = face1.normal.clone()
  const face2Normal = normalDot < 0 ? face2.normal.clone().negate() : face2.normal.clone()

  // Average normal for the projection plane
  const avgNormal = normal.clone().add(face2Normal).normalize()
  const { u, v } = createPlaneBasis(avgNormal)

  // Get combined vertices from all coplanar faces
  const vertices1 = getCombinedFaceVertices(face1Indices, allFaces, positionAttr)
  const vertices2 = getCombinedFaceVertices(face2Indices, allFaces, positionAttr)

  if (vertices1.length < 3 || vertices2.length < 3) return null

  // Use midpoint between combined face centers as plane origin
  const center1 = getCombinedFaceCenter(face1Indices, allFaces)
  const center2 = getCombinedFaceCenter(face2Indices, allFaces)
  const planeOrigin = center1.clone().add(center2).multiplyScalar(0.5)

  // Project vertices to 2D
  const projected1 = vertices1.map((v3) => projectToPlane(v3, planeOrigin, avgNormal, u, v))
  const projected2 = vertices2.map((v3) => projectToPlane(v3, planeOrigin, avgNormal, u, v))

  // Get convex hulls
  const hull1 = convexHull2D(projected1)
  const hull2 = convexHull2D(projected2)

  if (hull1.length < 3 || hull2.length < 3) return null

  // Find intersection of the two hulls
  const intersection = clipPolygon(hull1, hull2)

  if (intersection.length < 3) return null

  // Calculate plane distances using avgNormal as reference for consistent measurement
  const d1 = getFacePlaneDistance(face1, positionAttr, avgNormal)
  const d2 = getFacePlaneDistance(face2, positionAttr, avgNormal)
  const frontDist = Math.min(d1, d2)
  const backDist = Math.max(d1, d2)

  // Create 3D vertices from the 2D intersection polygon
  // Front face (closer to origin along normal)
  const frontVertices = intersection.map((p2d) => {
    const v3 = planeOrigin.clone()
    v3.add(u.clone().multiplyScalar(p2d.x))
    v3.add(v.clone().multiplyScalar(p2d.y))
    // Project to the front plane
    const currentDist = v3.dot(avgNormal)
    v3.add(avgNormal.clone().multiplyScalar(frontDist - currentDist))
    return v3
  })

  // Back face (further from origin along normal)
  const backVertices = intersection.map((p2d) => {
    const v3 = planeOrigin.clone()
    v3.add(u.clone().multiplyScalar(p2d.x))
    v3.add(v.clone().multiplyScalar(p2d.y))
    // Project to the back plane
    const currentDist = v3.dot(avgNormal)
    v3.add(avgNormal.clone().multiplyScalar(backDist - currentDist))
    return v3
  })

  const geometry = createExtrudedPolygonGeometry(frontVertices, backVertices)

  const wallData: WallData = {
    intersection,
    planeOrigin,
    uAxis: u,
    vAxis: v,
    drillAxis: avgNormal,
    drillStart: frontDist,
    drillEnd: backDist,
    face1Vertices: d1 <= d2 ? frontVertices : backVertices,
    face2Vertices: d1 <= d2 ? backVertices : frontVertices,
    face1Normal: face1.normal.clone(),
    face2Normal: face2.normal.clone(),
  }

  return { geometry, wallData }
}

// Create wall for faces with different normals (middle plane projection)
const createDifferentNormalWall = (
  face1Indices: number[],
  face2Indices: number[],
  allFaces: Face[],
  positionAttr: THREE.BufferAttribute,
): WallResult | null => {
  // Use the first face of each group as reference for normals
  const face1 = allFaces[face1Indices[0]!]!
  const face2 = allFaces[face2Indices[0]!]!

  // Find the bisector plane - the plane that has equal angles to both face normals
  // The bisector normal is the normalized sum of the two normals
  const bisectorNormal = face1.normal.clone().add(face2.normal).normalize()

  // If normals are nearly opposite, use cross product approach
  if (bisectorNormal.length() < 0.1) {
    // Normals are opposite - find a perpendicular plane
    const cross = new THREE.Vector3().crossVectors(face1.normal, face2.normal)
    if (cross.length() < 0.1) {
      // Truly opposite normals - use any perpendicular
      const { u } = createPlaneBasis(face1.normal)
      bisectorNormal.copy(u)
    } else {
      bisectorNormal.copy(cross).normalize()
    }
  }

  const { u, v } = createPlaneBasis(bisectorNormal)

  // Use midpoint between combined face centers as plane origin
  const center1 = getCombinedFaceCenter(face1Indices, allFaces)
  const center2 = getCombinedFaceCenter(face2Indices, allFaces)
  const planeOrigin = center1.clone().add(center2).multiplyScalar(0.5)

  // Get combined vertices from all coplanar faces
  const vertices1 = getCombinedFaceVertices(face1Indices, allFaces, positionAttr)
  const vertices2 = getCombinedFaceVertices(face2Indices, allFaces, positionAttr)

  if (vertices1.length < 3 || vertices2.length < 3) return null

  // Project vertices to the bisector plane
  const projected1 = vertices1.map((v3) => projectToPlane(v3, planeOrigin, bisectorNormal, u, v))
  const projected2 = vertices2.map((v3) => projectToPlane(v3, planeOrigin, bisectorNormal, u, v))

  // Get convex hulls
  const hull1 = convexHull2D(projected1)
  const hull2 = convexHull2D(projected2)

  if (hull1.length < 3 || hull2.length < 3) return null

  // Find intersection of the two hulls
  const intersection = clipPolygon(hull1, hull2)

  if (intersection.length < 3) return null

  // For different normals, we create a "slab" by projecting the intersection
  // back to each face's plane

  // Create vertices on face1's plane (use face1's normal for its own plane distance)
  const d1 = getFacePlaneDistance(face1, positionAttr, face1.normal)
  const vertices1_3d = intersection.map((p2d) => {
    const v3 = planeOrigin.clone()
    v3.add(u.clone().multiplyScalar(p2d.x))
    v3.add(v.clone().multiplyScalar(p2d.y))
    // Project along face1's normal to face1's plane
    const currentDist = v3.dot(face1.normal)
    v3.add(face1.normal.clone().multiplyScalar(d1 - currentDist))
    return v3
  })

  // Create vertices on face2's plane (use face2's normal for its own plane distance)
  const d2 = getFacePlaneDistance(face2, positionAttr, face2.normal)
  const vertices2_3d = intersection.map((p2d) => {
    const v3 = planeOrigin.clone()
    v3.add(u.clone().multiplyScalar(p2d.x))
    v3.add(v.clone().multiplyScalar(p2d.y))
    // Project along face2's normal to face2's plane
    const currentDist = v3.dot(face2.normal)
    v3.add(face2.normal.clone().multiplyScalar(d2 - currentDist))
    return v3
  })

  const geometry = createExtrudedPolygonGeometry(vertices1_3d, vertices2_3d)

  // For different normals, drill axis is perpendicular to the bisector plane
  // Calculate drill depths along bisector normal
  const drillDist1 = face1.center.dot(bisectorNormal)
  const drillDist2 = face2.center.dot(bisectorNormal)

  const wallData: WallData = {
    intersection,
    planeOrigin,
    uAxis: u,
    vAxis: v,
    drillAxis: bisectorNormal,
    drillStart: Math.min(drillDist1, drillDist2),
    drillEnd: Math.max(drillDist1, drillDist2),
    face1Vertices: vertices1_3d,
    face2Vertices: vertices2_3d,
    face1Normal: face1.normal.clone(),
    face2Normal: face2.normal.clone(),
  }

  return { geometry, wallData }
}

// Create flat polygon geometry from vertices (for overlay visualization)
const createFlatPolygonGeometry = (vertices: THREE.Vector3[]): THREE.BufferGeometry => {
  const positions: number[] = []
  const indices: number[] = []
  const n = vertices.length

  // Add vertices
  for (const v of vertices) {
    positions.push(v.x, v.y, v.z)
  }

  // Triangulate using fan from vertex 0
  for (let i = 1; i < n - 1; i++) {
    indices.push(0, i, i + 1)
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()

  return geometry
}

// Create geometry for an extruded polygon (connects two polygon faces)
const createExtrudedPolygonGeometry = (
  frontVertices: THREE.Vector3[],
  backVertices: THREE.Vector3[],
): THREE.BufferGeometry => {
  const positions: number[] = []
  const indices: number[] = []
  const n = frontVertices.length

  // Add front face vertices (0 to n-1)
  for (const v of frontVertices) {
    positions.push(v.x, v.y, v.z)
  }

  // Add back face vertices (n to 2n-1)
  for (const v of backVertices) {
    positions.push(v.x, v.y, v.z)
  }

  // Triangulate front face (fan from vertex 0)
  for (let i = 1; i < n - 1; i++) {
    indices.push(0, i, i + 1)
  }

  // Triangulate back face (fan from vertex n, reversed winding)
  for (let i = 1; i < n - 1; i++) {
    indices.push(n, n + i + 1, n + i)
  }

  // Side faces (quads split into triangles)
  for (let i = 0; i < n; i++) {
    const i1 = i
    const i2 = (i + 1) % n
    const i3 = n + i
    const i4 = n + ((i + 1) % n)

    // Two triangles per side quad
    indices.push(i1, i2, i4)
    indices.push(i1, i4, i3)
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()

  return geometry
}

// Clean up overlay meshes
const clearOverlayMeshes = () => {
  for (const mesh of overlayMeshes) {
    scene.remove(mesh)
    mesh.geometry.dispose()
    if (mesh.material instanceof THREE.Material) {
      mesh.material.dispose()
    }
  }
  overlayMeshes = []
}

// Create overlay mesh for a face showing the intersection region
const createOverlayMesh = (
  vertices: THREE.Vector3[],
  faceNormal: THREE.Vector3,
  scale: THREE.Vector3,
): THREE.Mesh => {
  const geometry = createFlatPolygonGeometry(vertices)

  const material = new THREE.MeshBasicMaterial({
    color: OVERLAP_COLOR,
    transparent: true,
    opacity: 0.7,
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  })

  const mesh = new THREE.Mesh(geometry, material)

  // Apply the same scale as the main mesh
  mesh.scale.copy(scale)

  return mesh
}

// Compute wall data based on selected faces (for drilling and overlap detection)
const computeWallData = () => {
  // Clear previous data and overlays
  clearOverlayMeshes()
  currentWallData = null

  // Only compute when exactly 2 faces are selected
  if (selectedFaces.value.length !== 2 || !currentMesh) return

  const selectedFace1 = selectedFaces.value[0]!
  const selectedFace2 = selectedFaces.value[1]!

  if (!faces.value[selectedFace1] || !faces.value[selectedFace2]) return

  const positionAttr = currentMesh.geometry.getAttribute('position') as THREE.BufferAttribute

  // Face 1 (first selected): use exactly the selected face - this determines the drill area
  // Face 2 (second selected): find connected coplanar faces - this is the "wall" which may have been
  // split by previous drilling operations (only faces sharing edges are included)
  const face1Indices = [selectedFace1]
  const coplanarFaces2 = findConnectedCoplanarFaces(selectedFace2, faces.value, positionAttr)

  const wallResult = createWallGeometry(face1Indices, coplanarFaces2, faces.value, positionAttr)

  if (!wallResult) return

  // Store wall data for drilling
  currentWallData = wallResult.wallData

  // Create overlay meshes for each face
  const { face1Vertices, face2Vertices, face1Normal, face2Normal } = currentWallData

  const overlay1 = createOverlayMesh(face1Vertices, face1Normal, currentMesh.scale)
  const overlay2 = createOverlayMesh(face2Vertices, face2Normal, currentMesh.scale)

  scene.add(overlay1)
  scene.add(overlay2)

  overlayMeshes.push(overlay1, overlay2)
}

// Calculate 2D bounding box of a polygon
const getPolygonBounds = (
  polygon: THREE.Vector2[],
): { minX: number; maxX: number; minY: number; maxY: number } => {
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity

  for (const p of polygon) {
    minX = Math.min(minX, p.x)
    maxX = Math.max(maxX, p.x)
    minY = Math.min(minY, p.y)
    maxY = Math.max(maxY, p.y)
  }

  return { minX, maxX, minY, maxY }
}

// Drill 3x3 holes through the wall
const drillHoles = () => {
  if (!currentMesh || !currentWallData) return

  isDrilling.value = true

  try {
    const { intersection, planeOrigin, uAxis, vAxis, drillAxis, drillStart, drillEnd } =
      currentWallData

    // Get bounds of the intersection polygon
    const bounds = getPolygonBounds(intersection)
    const width = bounds.maxX - bounds.minX
    const height = bounds.maxY - bounds.minY

    // Calculate hole radius based on available space
    // For a 3x3 grid with margin = padding = radius/2:
    // width = margin + 3*diameter + 2*padding = r/2 + 3*2r + 2*r/2 = r/2 + 6r + r = 7.5r
    // So radius = min(width, height) / 7.5
    const minDimension = Math.min(width, height)
    const radius = minDimension / 7.5
    const diameter = radius * 2
    const spacing = diameter + radius / 2 // diameter + padding

    // Center of the grid
    const centerX = (bounds.minX + bounds.maxX) / 2
    const centerY = (bounds.minY + bounds.maxY) / 2

    // Wall thickness plus extra for clean cuts
    const wallThickness = Math.abs(drillEnd - drillStart)
    const cylinderHeight = wallThickness * 2

    // Create CSG evaluator - only use position and normal (STL doesn't have UV)
    const evaluator = new Evaluator()
    evaluator.attributes = ['position', 'normal']

    // Prepare mesh geometry for CSG - needs to be indexed with only position and normal
    const meshGeometry = currentMesh.geometry.clone()

    // Remove all attributes except position and normal
    const attributeNames = Object.keys(meshGeometry.attributes)
    for (const name of attributeNames) {
      if (name !== 'position' && name !== 'normal') {
        meshGeometry.deleteAttribute(name)
      }
    }

    // Ensure we have normals
    if (!meshGeometry.hasAttribute('normal')) {
      meshGeometry.computeVertexNormals()
    }

    // Merge vertices to create indexed geometry (STL is non-indexed)
    const indexedGeometry = mergeVertices(meshGeometry)

    // Ensure geometry has an index (required for CSG)
    if (!indexedGeometry.index) {
      const positionAttr = indexedGeometry.getAttribute('position')
      const indices: number[] = []
      for (let i = 0; i < positionAttr.count; i++) {
        indices.push(i)
      }
      indexedGeometry.setIndex(indices)
    }

    // Recompute normals after merging
    indexedGeometry.computeVertexNormals()

    // CSG requires geometry groups - add a single group covering all triangles
    if (indexedGeometry.groups.length === 0) {
      indexedGeometry.addGroup(0, indexedGeometry.index!.count, 0)
    }

    // Create all hole brushes first, then subtract them all
    const holeBrushes: Brush[] = []

    // Calculate hole positions (3x3 grid, centered)
    const offsets = [-1, 0, 1]

    for (const row of offsets) {
      for (const col of offsets) {
        const holeX = centerX + col * spacing
        const holeY = centerY + row * spacing

        // Convert 2D position to 3D
        const holeCenter3D = planeOrigin.clone()
        holeCenter3D.add(uAxis.clone().multiplyScalar(holeX))
        holeCenter3D.add(vAxis.clone().multiplyScalar(holeY))

        // Create cylinder geometry for the hole
        const cylinderGeom = new THREE.CylinderGeometry(radius, radius, cylinderHeight, 32)

        // Remove all attributes except position and normal to match main geometry
        const cylAttrNames = Object.keys(cylinderGeom.attributes)
        for (const name of cylAttrNames) {
          if (name !== 'position' && name !== 'normal') {
            cylinderGeom.deleteAttribute(name)
          }
        }

        // Clear existing groups and add a single group
        cylinderGeom.clearGroups()
        cylinderGeom.addGroup(0, cylinderGeom.index!.count, 0)

        // Orient cylinder along drill axis
        // Default cylinder is along Y axis, need to rotate to align with drillAxis
        const defaultAxis = new THREE.Vector3(0, 1, 0)
        const quaternion = new THREE.Quaternion().setFromUnitVectors(defaultAxis, drillAxis)
        cylinderGeom.applyQuaternion(quaternion)

        // Position cylinder at hole center, centered on the wall
        const wallCenter = (drillStart + drillEnd) / 2
        const cylinderPosition = holeCenter3D
          .clone()
          .add(drillAxis.clone().multiplyScalar(wallCenter - holeCenter3D.dot(drillAxis)))
        cylinderGeom.translate(cylinderPosition.x, cylinderPosition.y, cylinderPosition.z)

        const holeBrush = new Brush(cylinderGeom)
        holeBrush.updateMatrixWorld()
        holeBrushes.push(holeBrush)
      }
    }

    // Create the main brush
    const mainBrush = new Brush(indexedGeometry)
    mainBrush.updateMatrixWorld()

    // Subtract first hole
    const firstHole = holeBrushes[0]!
    const resultBrush = evaluator.evaluate(mainBrush, firstHole, SUBTRACTION)

    // If first one works, do the rest
    let currentResult = resultBrush
    for (let i = 1; i < holeBrushes.length; i++) {
      const holeBrush = holeBrushes[i]!
      currentResult.updateMatrixWorld()
      const newResult = evaluator.evaluate(currentResult, holeBrush, SUBTRACTION)
      if (i > 1) {
        currentResult.geometry.dispose()
      }
      currentResult = newResult
    }

    const finalBrush = currentResult

    // Clean up hole brushes
    for (const holeBrush of holeBrushes) {
      holeBrush.geometry.dispose()
    }

    // Get the result geometry
    const resultGeometry = finalBrush.geometry

    // Remove old mesh
    scene.remove(currentMesh)
    currentMesh.geometry.dispose()
    if (currentMesh.material instanceof THREE.Material) {
      currentMesh.material.dispose()
    }

    // Re-analyze faces for the new geometry
    faces.value = analyzeFaces(resultGeometry)
    triangleToFaceMap = new Map()
    for (let i = 0; i < faces.value.length; i++) {
      for (const tri of faces.value[i]!.triangleIndices) {
        triangleToFaceMap.set(tri, i)
      }
    }

    // Set up vertex colors
    initVertexColors(resultGeometry)

    // Create new mesh with result
    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      specular: 0x222222,
      shininess: 50,
      flatShading: false,
      side: THREE.DoubleSide,
    })

    const newMesh = new THREE.Mesh(resultGeometry, material)
    newMesh.scale.copy(currentMesh.scale)

    currentMesh = newMesh
    scene.add(currentMesh)

    // Clear selection, wall data, and overlays
    selectedFaces.value = []
    currentWallData = null
    clearOverlayMeshes()
  } catch (err) {
    console.error('Failed to drill holes:', err)
    errorMessage.value = 'Failed to drill holes'
  } finally {
    isDrilling.value = false
  }
}

// Download the current mesh as an STL file
const downloadStl = () => {
  if (!currentMesh) return

  const exporter = new STLExporter()

  // Create a temporary mesh without scale for export (apply scale to geometry instead)
  const exportGeometry = currentMesh.geometry.clone()

  // Apply the mesh scale to the geometry so the exported STL has correct dimensions
  exportGeometry.scale(currentMesh.scale.x, currentMesh.scale.y, currentMesh.scale.z)

  const exportMesh = new THREE.Mesh(exportGeometry)

  // Export as binary STL (more compact)
  const stlData = exporter.parse(exportMesh, { binary: true })

  // Create blob and download
  const blob = new Blob([stlData], { type: 'application/octet-stream' })
  const url = URL.createObjectURL(blob)

  const link = document.createElement('a')
  link.href = url
  // Use original filename with "-drilled" suffix, or default name
  const baseName = fileName.value ? fileName.value.replace(/\.stl$/i, '') : 'model'
  link.download = `${baseName}-drilled.stl`
  link.click()

  URL.revokeObjectURL(url)

  // Clean up
  exportGeometry.dispose()
}

// Analyze geometry to extract faces using connected components with shared edges
const analyzeFaces = (geometry: THREE.BufferGeometry): Face[] => {
  const positionAttr = geometry.getAttribute('position') as THREE.BufferAttribute
  const normalAttr = geometry.getAttribute('normal') as THREE.BufferAttribute
  const triangleCount = positionAttr.count / 3
  const normalTolerance = 0.01

  // Build edge-to-triangles adjacency map
  const edgeToTriangles = new Map<string, number[]>()

  for (let tri = 0; tri < triangleCount; tri++) {
    const baseVertex = tri * 3
    const v0 = getVertex(positionAttr, baseVertex)
    const v1 = getVertex(positionAttr, baseVertex + 1)
    const v2 = getVertex(positionAttr, baseVertex + 2)

    // Three edges per triangle
    const edges = [makeEdgeKey(v0, v1), makeEdgeKey(v1, v2), makeEdgeKey(v2, v0)]

    for (const edge of edges) {
      if (!edgeToTriangles.has(edge)) {
        edgeToTriangles.set(edge, [])
      }
      edgeToTriangles.get(edge)!.push(tri)
    }
  }

  // Build triangle adjacency (neighbors that share an edge)
  const triangleNeighbors: number[][] = Array.from({ length: triangleCount }, () => [])

  edgeToTriangles.forEach((triangles) => {
    // Each edge typically has 1-2 triangles
    for (let i = 0; i < triangles.length; i++) {
      for (let j = i + 1; j < triangles.length; j++) {
        const triA = triangles[i]
        const triB = triangles[j]
        if (triA !== undefined && triB !== undefined) {
          triangleNeighbors[triA]?.push(triB)
          triangleNeighbors[triB]?.push(triA)
        }
      }
    }
  })

  // Flood-fill to find connected components with same normal
  const visited = new Set<number>()
  const extractedFaces: Face[] = []

  for (let startTri = 0; startTri < triangleCount; startTri++) {
    if (visited.has(startTri)) continue

    const faceNormal = getTriangleNormal(normalAttr, startTri)
    const faceTriangles: number[] = []
    const queue: number[] = [startTri]

    while (queue.length > 0) {
      const tri = queue.pop()!
      if (visited.has(tri)) continue

      const triNormal = getTriangleNormal(normalAttr, tri)
      if (faceNormal.distanceTo(triNormal) > normalTolerance) continue

      visited.add(tri)
      faceTriangles.push(tri)

      // Add unvisited neighbors with matching normal to queue
      const neighbors = triangleNeighbors[tri]
      if (neighbors) {
        for (const neighbor of neighbors) {
          if (!visited.has(neighbor)) {
            queue.push(neighbor)
          }
        }
      }
    }

    if (faceTriangles.length > 0) {
      // Compute face center from all vertices
      const center = new THREE.Vector3()
      let vertexCount = 0
      for (const tri of faceTriangles) {
        const baseVertex = tri * 3
        for (let v = 0; v < 3; v++) {
          center.add(getVertex(positionAttr, baseVertex + v))
          vertexCount++
        }
      }
      center.divideScalar(vertexCount)

      const faceIndex = extractedFaces.length
      extractedFaces.push({
        normal: faceNormal.clone(),
        triangleIndices: faceTriangles,
        center,
      })
      for (const tri of faceTriangles) {
        triangleToFaceMap.set(tri, faceIndex)
      }
    }
  }

  return extractedFaces
}

// Set up vertex colors for the geometry
const initVertexColors = (geometry: THREE.BufferGeometry) => {
  const positionCount = geometry.getAttribute('position').count
  const colors = new Float32Array(positionCount * 3)

  for (let i = 0; i < positionCount; i++) {
    colors[i * 3] = BASE_COLOR.r
    colors[i * 3 + 1] = BASE_COLOR.g
    colors[i * 3 + 2] = BASE_COLOR.b
  }

  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
}

// Highlight the selected faces and create overlay meshes for the overlap
const highlightSelectedFaces = () => {
  // Compute wall data and create overlay meshes (for 2 faces)
  computeWallData()

  if (!currentMesh) return

  const geometry = currentMesh.geometry
  const colorAttr = geometry.getAttribute('color')
  const positionAttr = geometry.getAttribute('position') as THREE.BufferAttribute
  if (!colorAttr) return

  // Reset all colors to base
  for (let i = 0; i < colorAttr.count; i++) {
    colorAttr.setXYZ(i, BASE_COLOR.r, BASE_COLOR.g, BASE_COLOR.b)
  }

  // Highlight selected faces
  // Use dim color when 2 faces selected (overlays show the precise overlap)
  // Use bright color when 1 face selected
  const highlightColor = selectedFaces.value.length === 2 ? DIM_SELECTED_COLOR : HIGHLIGHT_COLOR

  // For visual highlighting, include connected coplanar faces so the user can see the entire wall
  // Only faces that share edges AND are on the same plane are included
  const allFacesToHighlight = new Set<number>()
  for (const faceIndex of selectedFaces.value) {
    const connectedCoplanarFaces = findConnectedCoplanarFaces(faceIndex, faces.value, positionAttr)
    for (const idx of connectedCoplanarFaces) {
      allFacesToHighlight.add(idx)
    }
  }

  for (const faceIndex of allFacesToHighlight) {
    const face = faces.value[faceIndex]
    if (!face) continue

    for (const triangleIndex of face.triangleIndices) {
      const vertexStart = triangleIndex * 3
      for (let v = 0; v < 3; v++) {
        colorAttr.setXYZ(vertexStart + v, highlightColor.r, highlightColor.g, highlightColor.b)
      }
    }
  }

  colorAttr.needsUpdate = true
}

// Track mouse down position to distinguish clicks from drags
const handleCanvasMouseDown = (event: MouseEvent) => {
  mouseDownPosition = { x: event.clientX, y: event.clientY }
}

// Handle click on the canvas (only if it wasn't a drag)
const handleCanvasClick = (event: MouseEvent) => {
  if (!containerRef.value || !currentMesh) return

  // Check if this was a drag (mouse moved more than 5 pixels)
  if (mouseDownPosition) {
    const dx = event.clientX - mouseDownPosition.x
    const dy = event.clientY - mouseDownPosition.y
    const distance = Math.sqrt(dx * dx + dy * dy)
    if (distance > 5) {
      // This was a drag, not a click - ignore
      return
    }
  }

  const rect = containerRef.value.getBoundingClientRect()
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

  raycaster.setFromCamera(mouse, camera)
  const intersects = raycaster.intersectObject(currentMesh)

  if (intersects.length > 0) {
    const intersect = intersects[0]
    if (!intersect) return

    // faceIndex is the triangle index in the geometry
    const triangleIndex = intersect.faceIndex
    if (triangleIndex !== undefined && triangleIndex !== null) {
      const faceIndex = triangleToFaceMap.get(triangleIndex)
      if (faceIndex !== undefined) {
        if (event.metaKey || event.ctrlKey) {
          // Meta/Ctrl+click: add to selection or remove if already selected
          const existingIndex = selectedFaces.value.indexOf(faceIndex)
          if (existingIndex !== -1) {
            // Already selected - remove it
            selectedFaces.value.splice(existingIndex, 1)
          } else if (selectedFaces.value.length < 2) {
            // Add to selection (max 2 faces)
            selectedFaces.value.push(faceIndex)
          } else {
            // Already have 2 faces - replace the second one
            selectedFaces.value[1] = faceIndex
          }
        } else {
          // Regular click: select only this face
          selectedFaces.value = [faceIndex]
        }
        highlightSelectedFaces()
      }
    }
  } else {
    // Clicked outside the mesh - deselect all
    selectedFaces.value = []
    highlightSelectedFaces()
  }
}

const initScene = () => {
  if (!containerRef.value) return

  const width = containerRef.value.clientWidth
  const height = containerRef.value.clientHeight

  // Scene
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x1a1a2e)

  // Camera
  camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 2000)
  camera.position.set(0, 0, 100)

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setPixelRatio(window.devicePixelRatio)
  renderer.setSize(width, height)
  containerRef.value.appendChild(renderer.domElement)

  // Controls
  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.05

  // Raycaster for click detection
  raycaster = new THREE.Raycaster()
  mouse = new THREE.Vector2()

  // Lighting
  const ambientLight = new THREE.AmbientLight(0x404040, 2)
  scene.add(ambientLight)

  const directionalLight1 = new THREE.DirectionalLight(0xffffff, 1.5)
  directionalLight1.position.set(1, 1, 1)
  scene.add(directionalLight1)

  const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.8)
  directionalLight2.position.set(-1, -1, -1)
  scene.add(directionalLight2)

  // Grid helper for reference
  const gridHelper = new THREE.GridHelper(200, 20, 0x444466, 0x333355)
  gridHelper.rotation.x = Math.PI / 2
  scene.add(gridHelper)

  // Add mouse listeners for click detection (mousedown to track drag vs click)
  renderer.domElement.addEventListener('mousedown', handleCanvasMouseDown)
  renderer.domElement.addEventListener('click', handleCanvasClick)

  animate()
}

const animate = () => {
  animationFrameId = requestAnimationFrame(animate)
  controls.update()
  renderer.render(scene, camera)
}

const handleResize = () => {
  if (!containerRef.value) return
  const width = containerRef.value.clientWidth
  const height = containerRef.value.clientHeight
  camera.aspect = width / height
  camera.updateProjectionMatrix()
  renderer.setSize(width, height)
}

const loadStl = (file: File) => {
  isLoading.value = true
  errorMessage.value = null
  fileName.value = file.name

  // Reset face data
  faces.value = []
  selectedFaces.value = []
  triangleToFaceMap = new Map()
  currentWallData = null
  clearOverlayMeshes()

  const reader = new FileReader()

  reader.onload = (event) => {
    if (!event.target?.result) return

    try {
      const loader = new STLLoader()
      const geometry = loader.parse(event.target.result as ArrayBuffer)

      // Remove existing mesh
      if (currentMesh) {
        scene.remove(currentMesh)
        currentMesh.geometry.dispose()
        if (currentMesh.material instanceof THREE.Material) {
          currentMesh.material.dispose()
        }
      }

      // Analyze faces before modifying geometry
      faces.value = analyzeFaces(geometry)

      // Set up vertex colors for highlighting
      initVertexColors(geometry)

      // Create material that uses vertex colors
      // side: DoubleSide renders both front and back of triangles
      const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        specular: 0x222222,
        shininess: 50,
        flatShading: false,
        side: THREE.DoubleSide,
      })

      currentMesh = new THREE.Mesh(geometry, material)

      // Center the geometry
      geometry.computeBoundingBox()
      const boundingBox = geometry.boundingBox!
      const center = new THREE.Vector3()
      boundingBox.getCenter(center)
      geometry.translate(-center.x, -center.y, -center.z)

      // Scale to fit viewport
      const size = new THREE.Vector3()
      boundingBox.getSize(size)
      const maxDim = Math.max(size.x, size.y, size.z)
      const scale = 80 / maxDim
      currentMesh.scale.set(scale, scale, scale)

      scene.add(currentMesh)

      // Reset camera position
      camera.position.set(0, 0, 100)
      controls.reset()

      isLoading.value = false
    } catch (err) {
      errorMessage.value = 'Failed to parse STL file'
      isLoading.value = false
      console.error(err)
    }
  }

  reader.onerror = () => {
    errorMessage.value = 'Failed to read file'
    isLoading.value = false
  }

  reader.readAsArrayBuffer(file)
}

const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    loadStl(file)
  }
}

const triggerFileInput = () => {
  fileInputRef.value?.click()
}

const handleDrop = (event: DragEvent) => {
  event.preventDefault()
  const file = event.dataTransfer?.files[0]
  if (file && file.name.toLowerCase().endsWith('.stl')) {
    loadStl(file)
  } else if (file) {
    errorMessage.value = 'Please drop an STL file'
  }
}

const handleDragOver = (event: DragEvent) => {
  event.preventDefault()
}

onMounted(() => {
  initScene()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  cancelAnimationFrame(animationFrameId)

  if (renderer?.domElement) {
    renderer.domElement.removeEventListener('mousedown', handleCanvasMouseDown)
    renderer.domElement.removeEventListener('click', handleCanvasClick)
  }

  if (currentMesh) {
    currentMesh.geometry.dispose()
    if (currentMesh.material instanceof THREE.Material) {
      currentMesh.material.dispose()
    }
  }

  clearOverlayMeshes()
  renderer?.dispose()
})
</script>

<template>
  <div class="stl-viewer">
    <div class="controls">
      <input
        ref="fileInputRef"
        type="file"
        accept=".stl"
        @change="handleFileSelect"
        class="file-input"
      />
      <button @click="triggerFileInput" class="load-button">
        {{ fileName ? 'Load Different STL' : 'Load STL File' }}
      </button>
      <button v-if="fileName" @click="downloadStl" class="download-button">Download STL</button>
      <span v-if="fileName" class="file-name">{{ fileName }}</span>
    </div>

    <div v-if="errorMessage" class="error">{{ errorMessage }}</div>

    <div v-if="faces.length > 0" class="face-info">
      <div class="face-count">
        <span class="label">Faces:</span>
        <span class="value">{{ faces.length }}</span>
      </div>
      <div v-if="selectedFaces.length > 0" class="selected-face">
        <span class="label">Selected:</span>
        <span class="value">
          Face {{ selectedFaces[0]! + 1 }}
          <template v-if="selectedFaces.length === 2">
            &amp; Face {{ selectedFaces[1]! + 1 }}
          </template>
        </span>
      </div>
      <div v-else class="selection-hint">
        Click to select a face, ⌘/Ctrl+click to select the second face
      </div>
      <button
        v-if="selectedFaces.length === 2"
        @click="drillHoles"
        :disabled="isDrilling || !canDrill"
        class="drill-button"
      >
        {{ isDrilling ? 'Drilling...' : 'Drill 3×3 Holes' }}
      </button>
    </div>

    <div ref="containerRef" class="canvas-container" @drop="handleDrop" @dragover="handleDragOver">
      <div v-if="isLoading" class="loading">Loading...</div>
      <div v-if="!fileName && !isLoading" class="drop-hint">
        Drop an STL file here or click the button above
      </div>
    </div>
  </div>
</template>

<style scoped>
.stl-viewer {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  width: 100%;
  max-width: 800px;
}

.controls {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}

.file-input {
  display: none;
}

.load-button {
  background: linear-gradient(135deg, #00d4aa 0%, #00a88a 100%);
  color: #1a1a2e;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition:
    transform 0.2s,
    box-shadow 0.2s;
}

.load-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(0, 212, 170, 0.4);
}

.load-button:active {
  transform: translateY(0);
}

.download-button {
  background: linear-gradient(135deg, #6b8dd6 0%, #4a6cb3 100%);
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition:
    transform 0.2s,
    box-shadow 0.2s;
}

.download-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(107, 141, 214, 0.4);
}

.download-button:active {
  transform: translateY(0);
}

.file-name {
  color: #888;
  font-size: 0.9rem;
  font-style: italic;
}

.error {
  color: #ff6b6b;
  background: rgba(255, 107, 107, 0.1);
  padding: 0.75rem 1rem;
  border-radius: 8px;
  border: 1px solid rgba(255, 107, 107, 0.3);
}

.canvas-container {
  width: 100%;
  height: 500px;
  border-radius: 12px;
  overflow: hidden;
  position: relative;
  background: #1a1a2e;
  border: 2px dashed #333355;
  transition: border-color 0.2s;
}

.canvas-container:hover {
  border-color: #00d4aa;
}

.loading,
.drop-hint {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: #666;
  font-size: 1.1rem;
  text-align: center;
  pointer-events: none;
}

.drop-hint {
  max-width: 250px;
  line-height: 1.6;
}

.face-info {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  padding: 0.75rem 1rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  font-size: 0.9rem;
}

.face-count,
.selected-face {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.face-info .label {
  color: #888;
}

.face-info .value {
  color: #00d4aa;
  font-weight: 600;
}

.selected-face .value {
  color: #ff6b6b;
}

.triangle-count {
  color: #666;
  font-size: 0.85rem;
}

.selection-hint {
  color: #666;
  font-style: italic;
}

.drill-button {
  background: linear-gradient(135deg, #4488ff 0%, #2266dd 100%);
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  transition:
    transform 0.2s,
    box-shadow 0.2s,
    opacity 0.2s;
  margin-left: auto;
}

.drill-button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 16px rgba(68, 136, 255, 0.4);
}

.drill-button:active:not(:disabled) {
  transform: translateY(0);
}

.drill-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
