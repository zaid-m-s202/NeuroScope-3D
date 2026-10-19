/**
 * NeuroScope 3D - Spatial Coordinate Transformation Tests
 * ========================================================
 * Validates:
 * 1. Normal affine transforms (standard isotropic & translations)
 * 2. Axis reflections (e.g. LPS space with negative X/Y axes)
 * 3. Non-isotropic voxel spacing (e.g. anisotropic slice thickness)
 * 4. Invalid, singular, null, and NaN-containing affine handling
 * 5. Round-trip conversions between voxel and world space
 */
import { describe, it, expect } from 'vitest';
import {
  normalizeAffine,
  voxelToWorld,
  worldToVoxel,
  computeVoxelSpacing,
  computeVoxelVolume,
  getAnatomicalOrientation,
  matrixInverse4x4,
} from '../utils/coordinates';

describe('Spatial Coordinate & Affine Transformations', () => {
  describe('1. Normal Affine Transforms', () => {
    it('correctly maps discrete voxel indices through an isotropic RAS affine with translation', () => {
      // 1.0 mm isotropic spacing with origin translation [-100, -100, -50]
      const affine = [
        [1.0, 0.0, 0.0, -100.0],
        [0.0, 1.0, 0.0, -100.0],
        [0.0, 0.0, 1.0, -50.0],
        [0.0, 0.0, 0.0, 1.0],
      ];

      const vox = [100, 150, 75];
      const world = voxelToWorld(vox, affine);

      expect(world).toEqual([0.0, 50.0, 25.0]);
      expect(Number.isFinite(world[0])).toBe(true);
      expect(Number.isFinite(world[1])).toBe(true);
      expect(Number.isFinite(world[2])).toBe(true);

      // Round-trip check
      const roundTripVox = worldToVoxel(world, affine);
      expect(roundTripVox).toEqual(vox);
    });

    it('handles flat 16-element column-major WebGL/Niivue matrices', () => {
      // Column-major representation of translation [10, 20, 30] with 1mm spacing
      const flatColMajor = [
        1, 0, 0, 0,  // col 0
        0, 1, 0, 0,  // col 1
        0, 0, 1, 0,  // col 2
        10, 20, 30, 1 // col 3
      ];

      const vox = [5, 5, 5];
      const world = voxelToWorld(vox, flatColMajor);
      expect(world).toEqual([15.0, 25.0, 35.0]);
    });
  });

  describe('2. Axis Reflections (e.g. LPS and Radiological Coordinates)', () => {
    it('correctly computes world coordinates under LPS reflections (negative diagonals)', () => {
      // BraTS standard orientation: LPS with negative X and Y scales and offset
      const lpsAffine = [
        [-1.0,  0.0, 0.0,   0.0],
        [ 0.0, -1.0, 0.0, 239.0],
        [ 0.0,  0.0, 1.0,   0.0],
        [ 0.0,  0.0, 0.0,   1.0],
      ];

      // Origin voxel
      expect(voxelToWorld([0, 0, 0], lpsAffine)).toEqual([0.0, 239.0, 0.0]);

      // Mid-brain voxel
      expect(voxelToWorld([120, 120, 78], lpsAffine)).toEqual([-120.0, 119.0, 78.0]);

      // Max voxel extent [239, 239, 154]
      expect(voxelToWorld([239, 239, 154], lpsAffine)).toEqual([-239.0, 0.0, 154.0]);

      // Orientation detection
      expect(getAnatomicalOrientation(lpsAffine)).toBe('LPS');
    });

    it('inverts reflected affines properly back to discrete voxel space', () => {
      const lpsAffine = [
        [-1.0,  0.0, 0.0,   0.0],
        [ 0.0, -1.0, 0.0, 239.0],
        [ 0.0,  0.0, 1.0,   0.0],
        [ 0.0,  0.0, 0.0,   1.0],
      ];

      const world = [-98.27, 74.59, 98.14];
      const vox = worldToVoxel(world, lpsAffine);

      // (-98.27) * -1 = 98.27 -> 98
      // (74.59 - 239.0) * -1 = 164.41 -> 164
      // 98.14 -> 98
      expect(vox).toEqual([98, 164, 98]);
    });
  });

  describe('3. Non-Isotropic Voxel Spacing', () => {
    it('scales coordinates accurately with anisotropic voxel dimensions', () => {
      // Clinical anisotropic scan: 0.5mm in-plane, 3.0mm slice thickness
      const anisotropicAffine = [
        [0.5, 0.0, 0.0,  -50.0],
        [0.0, 0.5, 0.0,  -50.0],
        [0.0, 0.0, 3.0, -100.0],
        [0.0, 0.0, 0.0,    1.0],
      ];

      const vox = [100, 200, 40];
      const world = voxelToWorld(vox, anisotropicAffine);

      // x: 0.5 * 100 - 50 = 0.0
      // y: 0.5 * 200 - 50 = 50.0
      // z: 3.0 * 40 - 100 = 20.0
      expect(world).toEqual([0.0, 50.0, 20.0]);

      // Voxel spacing verification
      const spacing = computeVoxelSpacing(anisotropicAffine);
      expect(spacing).toEqual([0.5, 0.5, 3.0]);

      // Voxel volume calculation (0.5 * 0.5 * 3.0 = 0.75 mm³, 0.00075 mL)
      const vol = computeVoxelVolume(anisotropicAffine);
      expect(vol.mm3).toBe(0.75);
      expect(vol.ml).toBe(0.00075);
    });
  });

  describe('4. Missing, Invalid, and Degenerate Affine Handling', () => {
    it('returns finite world coordinates when affine is null, undefined, or empty', () => {
      const fallbackSpacing = [2.0, 2.0, 2.0];

      // Null affine falls back to diagonal matrix with fallback spacing
      const worldFromNull = voxelToWorld([10, 20, 30], null, fallbackSpacing);
      expect(worldFromNull).toEqual([20.0, 40.0, 60.0]);
      expect(worldFromNull.every((v) => Number.isFinite(v))).toBe(true);

      // Undefined affine falls back to identity [1, 1, 1]
      const worldFromUndef = voxelToWorld([10, 20, 30], undefined);
      expect(worldFromUndef).toEqual([10.0, 20.0, 30.0]);
      expect(worldFromUndef.every((v) => Number.isFinite(v))).toBe(true);
    });

    it('safely handles matrices containing NaN or Infinity without propagating NaN', () => {
      const corruptedAffine = [
        [NaN, 0.0, 0.0, 10.0],
        [0.0, Infinity, 0.0, 20.0],
        [0.0, 0.0, null, 30.0],
        [0.0, 0.0, 0.0, 1.0],
      ];

      const world = voxelToWorld([5, 10, 15], corruptedAffine, [1.0, 1.0, 1.0]);
      expect(world.every((v) => Number.isFinite(v))).toBe(true);
      expect(world.some((v) => isNaN(v))).toBe(false);
    });

    it('safely handles singular (zero determinant) affine matrices during inversion', () => {
      // Degenerate/singular matrix (all zeros in third row)
      const singularAffine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0], // singular
        [0.0, 0.0, 0.0, 1.0],
      ];

      // matrixInverse4x4 throws a descriptive error
      expect(() => matrixInverse4x4(singularAffine)).toThrow('Singular affine matrix cannot be inverted');

      // worldToVoxel catches this error and safely falls back without throwing or NaN
      const vox = worldToVoxel([15, 25, 35], singularAffine, [100, 100, 100]);
      expect(Array.isArray(vox)).toBe(true);
      expect(vox.every((v) => Number.isFinite(v))).toBe(true);
      expect(vox.some((v) => isNaN(v))).toBe(false);
    });

    it('safely handles non-finite voxel coordinates input', () => {
      const affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
      ];

      // NaN in voxel coordinates
      const worldFromNaN = voxelToWorld([NaN, 10, undefined], affine);
      expect(worldFromNaN.every((v) => Number.isFinite(v))).toBe(true);
      expect(worldFromNaN).toEqual([0.0, 10.0, 0.0]);
    });
  });
});
