// Types for hole grid computation (independent of Three.js)

export type Point2D = {
  x: number
  y: number
}

export type Bounds2D = {
  minX: number
  maxX: number
  minY: number
  maxY: number
}

export type HoleGridResult = {
  positions: Point2D[]
  numX: number
  numY: number
}

/**
 * Calculate 2D bounding box of a polygon
 */
export const getPolygonBounds = (polygon: Point2D[]): Bounds2D => {
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

/**
 * Calculate the hole grid layout based on diameter and available space.
 * Returns null if no holes can fit.
 *
 * The algorithm:
 * 1. Calculates minimum gap between holes and from edges (at least 20% of diameter, minimum 0.5mm)
 * 2. Determines how many holes fit in each direction using: N <= (dimension - gap) / (diameter + gap)
 * 3. Distributes gaps evenly so holes are centered in the available space
 * 4. Returns grid positions, count in X direction, and count in Y direction
 */
export const calculateHoleGrid = (bounds: Bounds2D, diameter: number): HoleGridResult | null => {
  const width = bounds.maxX - bounds.minX
  const height = bounds.maxY - bounds.minY
  const radius = diameter / 2

  // Minimum gap between holes and from edges (at least 20% of diameter, minimum 0.5mm)
  const minGap = Math.max(0.5, diameter * 0.2)

  // Calculate how many holes fit in each direction
  // Formula: N holes need N*D space for holes + (N+1)*gap for spacing
  // So: N*D + (N+1)*gap <= dimension
  // N*(D + gap) <= dimension - gap
  // N <= (dimension - gap) / (D + gap)
  const numX = Math.max(1, Math.floor((width - minGap) / (diameter + minGap)))
  const numY = Math.max(1, Math.floor((height - minGap) / (diameter + minGap)))

  // Check if at least one hole fits (need space for diameter + 2 * edge gap)
  if (width < diameter + 2 * minGap || height < diameter + 2 * minGap) {
    return null
  }

  // Calculate actual gaps to distribute evenly
  // Total space used: numX * diameter + (numX + 1) * gapX = width
  // gapX = (width - numX * diameter) / (numX + 1)
  const gapX = (width - numX * diameter) / (numX + 1)
  const gapY = (height - numY * diameter) / (numY + 1)

  // Generate hole positions
  // First hole center: bounds.min + gap + radius
  // Subsequent holes: spaced by (diameter + gap)
  const positions: Point2D[] = []

  for (let row = 0; row < numY; row++) {
    for (let col = 0; col < numX; col++) {
      const x = bounds.minX + gapX + radius + col * (diameter + gapX)
      const y = bounds.minY + gapY + radius + row * (diameter + gapY)
      positions.push({ x, y })
    }
  }

  return { positions, numX, numY }
}
