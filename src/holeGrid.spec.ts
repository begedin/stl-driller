import { describe, it, expect } from 'vitest'
import { getPolygonBounds, calculateHoleGrid } from './holeGrid'
import type { Point2D, Bounds2D } from './holeGrid'

describe('getPolygonBounds', () => {
  it('returns correct bounds for a simple rectangle', () => {
    const polygon: Point2D[] = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 5 },
      { x: 0, y: 5 },
    ]

    const bounds = getPolygonBounds(polygon)

    expect(bounds).toEqual({ minX: 0, maxX: 10, minY: 0, maxY: 5 })
  })

  it('handles negative coordinates', () => {
    const polygon: Point2D[] = [
      { x: -5, y: -3 },
      { x: 5, y: -3 },
      { x: 5, y: 3 },
      { x: -5, y: 3 },
    ]

    const bounds = getPolygonBounds(polygon)

    expect(bounds).toEqual({ minX: -5, maxX: 5, minY: -3, maxY: 3 })
  })

  it('handles a triangle', () => {
    const polygon: Point2D[] = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 5, y: 8 },
    ]

    const bounds = getPolygonBounds(polygon)

    expect(bounds).toEqual({ minX: 0, maxX: 10, minY: 0, maxY: 8 })
  })

  it('handles a single point', () => {
    const polygon: Point2D[] = [{ x: 3, y: 7 }]

    const bounds = getPolygonBounds(polygon)

    expect(bounds).toEqual({ minX: 3, maxX: 3, minY: 7, maxY: 7 })
  })
})

describe('calculateHoleGrid', () => {
  it('returns null when area is too small for the specified hole diameter', () => {
    const bounds: Bounds2D = { minX: 0, maxX: 3, minY: 0, maxY: 3 }
    // 5mm diameter with 1mm min gap (20% of 5) needs at least 7mm (5 + 2*1)
    const result = calculateHoleGrid(bounds, 5)

    expect(result).toBeNull()
  })

  it('fits a single hole when there is just enough space', () => {
    // 5mm diameter has 20% = 1mm min gap, so needs 5 + 2*1 = 7mm minimum
    const bounds: Bounds2D = { minX: 0, maxX: 7, minY: 0, maxY: 7 }

    const result = calculateHoleGrid(bounds, 5)

    expect(result).not.toBeNull()
    expect(result!.numX).toBe(1)
    expect(result!.numY).toBe(1)
    expect(result!.positions).toHaveLength(1)
  })

  it('calculates correct grid for larger area', () => {
    // 20x20 area with 5mm holes (1mm min gap)
    // Available: 20mm, need per hole: 5 + 1 = 6mm
    // N = floor((20 - 1) / 6) = floor(19/6) = 3 holes per direction
    const bounds: Bounds2D = { minX: 0, maxX: 20, minY: 0, maxY: 20 }

    const result = calculateHoleGrid(bounds, 5)

    expect(result).not.toBeNull()
    expect(result!.numX).toBe(3)
    expect(result!.numY).toBe(3)
    expect(result!.positions).toHaveLength(9)
  })

  it('positions holes centered in the available space', () => {
    // 7mm width with 5mm hole (1mm min gap): 1 hole
    // Gap = (7 - 5*1) / (1+1) = 2/2 = 1mm on each side
    // Center = 0 + 1 + 2.5 = 3.5
    const bounds: Bounds2D = { minX: 0, maxX: 7, minY: 0, maxY: 7 }

    const result = calculateHoleGrid(bounds, 5)

    expect(result!.positions[0]!.x).toBe(3.5)
    expect(result!.positions[0]!.y).toBe(3.5)
  })

  it('handles non-square bounds', () => {
    // 20mm wide, 10mm tall with 5mm holes (1mm min gap)
    // Width: floor((20-1)/6) = 3 holes
    // Height: floor((10-1)/6) = 1 hole
    const bounds: Bounds2D = { minX: 0, maxX: 20, minY: 0, maxY: 10 }

    const result = calculateHoleGrid(bounds, 5)

    expect(result).not.toBeNull()
    expect(result!.numX).toBe(3)
    expect(result!.numY).toBe(1)
    expect(result!.positions).toHaveLength(3)
  })

  it('uses 0.5mm minimum gap for very small hole diameters', () => {
    // For 1mm diameter, 20% = 0.2mm, but min is 0.5mm
    // So needs 1 + 2*0.5 = 2mm minimum space
    const bounds: Bounds2D = { minX: 0, maxX: 1.9, minY: 0, maxY: 1.9 }

    const result = calculateHoleGrid(bounds, 1)

    expect(result).toBeNull()
  })

  it('fits small holes when there is enough space with 0.5mm min gap', () => {
    // 1mm diameter, 0.5mm min gap, needs 2mm minimum
    const bounds: Bounds2D = { minX: 0, maxX: 2, minY: 0, maxY: 2 }

    const result = calculateHoleGrid(bounds, 1)

    expect(result).not.toBeNull()
    expect(result!.numX).toBe(1)
    expect(result!.numY).toBe(1)
  })

  it('handles bounds with non-zero origin', () => {
    const bounds: Bounds2D = { minX: 10, maxX: 30, minY: 5, maxY: 25 }

    const result = calculateHoleGrid(bounds, 5)

    expect(result).not.toBeNull()
    // All positions should be within bounds
    for (const pos of result!.positions) {
      expect(pos.x).toBeGreaterThan(bounds.minX)
      expect(pos.x).toBeLessThan(bounds.maxX)
      expect(pos.y).toBeGreaterThan(bounds.minY)
      expect(pos.y).toBeLessThan(bounds.maxY)
    }
  })

  it('returns positions in row-major order', () => {
    const bounds: Bounds2D = { minX: 0, maxX: 20, minY: 0, maxY: 20 }

    const result = calculateHoleGrid(bounds, 5)

    // Should iterate rows then cols: (0,0), (1,0), (2,0), (0,1), ...
    // First row should all have same y
    const firstRowY = result!.positions[0]!.y
    expect(result!.positions[1]!.y).toBe(firstRowY)
    expect(result!.positions[2]!.y).toBe(firstRowY)
    // Second row should have different y
    expect(result!.positions[3]!.y).toBeGreaterThan(firstRowY)
  })
})
