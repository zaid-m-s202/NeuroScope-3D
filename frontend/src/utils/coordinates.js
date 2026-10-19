/**
 * NeuroScope 3D - Spatial Coordinate & Affine Utilities
 * ======================================================
 * Mathematically rigorous coordinate conversions between discrete voxel grid [i, j, k]
 * and continuous physical world millimetre [x, y, z] (RAS/LPS space) using NIfTI 4x4 spatial affines.
 *
 * Guarantees:
 * - Valid voxel coordinates always produce finite world coordinates.
 * - Flat 16-element matrices (Float32Array / column-major WebGL) and 4x4 2D arrays are supported.
 * - Missing, null, singular, or corrupted affine matrices (containing NaN/Inf) fall back safely
 *   to voxel spacing diagonal scaling without producing NaN.
 * - Axis reflections (e.g. LPS negative diagonals) and non-isotropic spacing are preserved.
 */

/**
 * Normalizes an arbitrary affine representation into a guaranteed valid 4x4 finite number matrix.
 * Supports:
 * - 4x4 array of arrays: [[...], [...], [...], [...]]
 * - 16-element 1D array or Float32Array (column-major glMatrix / Niivue format or row-major)
 * - Fallback to diagonal matrix based on physical voxel spacing.
 *
 * @param {number[][] | ArrayLike<number> | null | undefined} affine
 * @param {[number, number, number]} [fallbackSpacing=[1.0, 1.0, 1.0]]
 * @returns {number[][]} guaranteed finite 4x4 affine matrix
 */
export function normalizeAffine(affine, fallbackSpacing = [1.0, 1.0, 1.0]) {
  const sx = Number.isFinite(fallbackSpacing?.[0]) && fallbackSpacing[0] > 0 ? fallbackSpacing[0] : 1.0;
  const sy = Number.isFinite(fallbackSpacing?.[1]) && fallbackSpacing[1] > 0 ? fallbackSpacing[1] : 1.0;
  const sz = Number.isFinite(fallbackSpacing?.[2]) && fallbackSpacing[2] > 0 ? fallbackSpacing[2] : 1.0;

  const defaultMatrix = [
    [sx, 0.0, 0.0, 0.0],
    [0.0, sy, 0.0, 0.0],
    [0.0, 0.0, sz, 0.0],
    [0.0, 0.0, 0.0, 1.0],
  ];

  if (!affine) return defaultMatrix;

  // Handle 1D array of 16 numbers (Float32Array or flat array)
  if ((Array.isArray(affine) || ArrayBuffer.isView(affine)) && affine.length === 16) {
    // Check if column-major (Niivue matRAS convention: translation at indices 12, 13, 14, last row [0,0,0,1])
    const isColMajor = (
      Math.abs(affine[3]) < 1e-5 &&
      Math.abs(affine[7]) < 1e-5 &&
      Math.abs(affine[11]) < 1e-5 &&
      Math.abs(affine[15] - 1.0) < 1e-5
    );

    const m = Array.from({ length: 4 }, () => new Array(4).fill(0));
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        const val = isColMajor ? affine[c * 4 + r] : affine[r * 4 + c];
        if (!Number.isFinite(val)) return defaultMatrix;
        m[r][c] = Number(val);
      }
    }
    return m;
  }

  // Handle 4x4 array of arrays
  if (Array.isArray(affine) && affine.length >= 4) {
    const m = Array.from({ length: 4 }, () => new Array(4).fill(0));
    for (let r = 0; r < 4; r++) {
      if (!Array.isArray(affine[r]) && !ArrayBuffer.isView(affine[r])) return defaultMatrix;
      if (affine[r].length < 4) return defaultMatrix;
      for (let c = 0; c < 4; c++) {
        const val = affine[r][c];
        if (!Number.isFinite(val)) return defaultMatrix;
        m[r][c] = Number(val);
      }
    }
    return m;
  }

  return defaultMatrix;
}

/**
 * Inverts a 4x4 affine transformation matrix.
 * @param {number[][]} m - 4x4 matrix
 * @returns {number[][]} inverse 4x4 matrix
 * @throws {Error} if matrix is singular or non-finite
 */
export function matrixInverse4x4(m) {
  const norm = normalizeAffine(m);
  const inv = Array.from({ length: 4 }, () => new Array(4).fill(0));

  const a00 = norm[0][0], a01 = norm[0][1], a02 = norm[0][2], a03 = norm[0][3];
  const a10 = norm[1][0], a11 = norm[1][1], a12 = norm[1][2], a13 = norm[1][3];
  const a20 = norm[2][0], a21 = norm[2][1], a22 = norm[2][2], a23 = norm[2][3];
  const a30 = norm[3][0], a31 = norm[3][1], a32 = norm[3][2], a33 = norm[3][3];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) {
    throw new Error('Singular affine matrix cannot be inverted');
  }
  const invDet = 1.0 / det;

  inv[0][0] = (a11 * b11 - a12 * b10 + a13 * b09) * invDet;
  inv[0][1] = (-a01 * b11 + a02 * b10 - a03 * b09) * invDet;
  inv[0][2] = (a31 * b05 - a32 * b04 + a33 * b03) * invDet;
  inv[0][3] = (-a21 * b05 + a22 * b04 - a23 * b03) * invDet;

  inv[1][0] = (-a10 * b11 + a12 * b08 - a13 * b07) * invDet;
  inv[1][1] = (a00 * b11 - a02 * b08 + a03 * b07) * invDet;
  inv[1][2] = (-a30 * b05 + a32 * b02 - a33 * b01) * invDet;
  inv[1][3] = (a20 * b05 - a22 * b02 + a23 * b01) * invDet;

  inv[2][0] = (a10 * b10 - a11 * b08 + a13 * b06) * invDet;
  inv[2][1] = (-a00 * b10 + a01 * b08 - a03 * b06) * invDet;
  inv[2][2] = (a30 * b04 - a31 * b02 + a33 * b00) * invDet;
  inv[2][3] = (-a20 * b04 + a21 * b02 - a23 * b00) * invDet;

  inv[3][0] = (-a10 * b09 + a11 * b07 - a12 * b06) * invDet;
  inv[3][1] = (a00 * b09 - a01 * b07 + a02 * b06) * invDet;
  inv[3][2] = (-a30 * b03 + a31 * b01 - a32 * b00) * invDet;
  inv[3][3] = (a20 * b03 - a21 * b01 + a22 * b00) * invDet;

  return inv;
}

/**
 * Transforms discrete voxel coordinates [i, j, k] to continuous physical world millimetres [x, y, z].
 * Formula: [x, y, z, 1]^T = Affine * [i, j, k, 1]^T
 * Guarantees that valid voxel coordinates always produce finite numbers.
 *
 * @param {[number, number, number]} voxel - [i, j, k] discrete voxel coordinates
 * @param {number[][] | ArrayLike<number> | null} affine - 4x4 NIfTI affine matrix
 * @param {[number, number, number]} [fallbackSpacing=[1.0, 1.0, 1.0]] - fallback voxel spacing
 * @returns {[number, number, number]} [x, y, z] in mm
 */
export function voxelToWorld(voxel, affine, fallbackSpacing = [1.0, 1.0, 1.0]) {
  const i = (voxel && Number.isFinite(voxel[0])) ? Number(voxel[0]) : 0.0;
  const j = (voxel && Number.isFinite(voxel[1])) ? Number(voxel[1]) : 0.0;
  const k = (voxel && Number.isFinite(voxel[2])) ? Number(voxel[2]) : 0.0;

  const A = normalizeAffine(affine, fallbackSpacing);

  const x = A[0][0] * i + A[0][1] * j + A[0][2] * k + A[0][3];
  const y = A[1][0] * i + A[1][1] * j + A[1][2] * k + A[1][3];
  const z = A[2][0] * i + A[2][1] * j + A[2][2] * k + A[2][3];

  const fx = Number.isFinite(x) ? Number(x.toFixed(2)) : 0.0;
  const fy = Number.isFinite(y) ? Number(y.toFixed(2)) : 0.0;
  const fz = Number.isFinite(z) ? Number(z.toFixed(2)) : 0.0;

  return [fx, fy, fz];
}

/**
 * Transforms continuous physical world coordinates [x, y, z] in mm to discrete voxel index [i, j, k].
 * Formula: [i, j, k, 1]^T = Affine^-1 * [x, y, z, 1]^T
 *
 * @param {[number, number, number]} world - [x, y, z] in mm
 * @param {number[][] | ArrayLike<number> | null} affine - 4x4 NIfTI affine matrix
 * @param {[number, number, number]} [bounds] - optional volume dimensions [dimX, dimY, dimZ] to clamp
 * @param {[number, number, number]} [fallbackSpacing=[1.0, 1.0, 1.0]]
 * @returns {[number, number, number]} [i, j, k] discrete voxel indices
 */
export function worldToVoxel(world, affine, bounds = null, fallbackSpacing = [1.0, 1.0, 1.0]) {
  const x = (world && Number.isFinite(world[0])) ? Number(world[0]) : 0.0;
  const y = (world && Number.isFinite(world[1])) ? Number(world[1]) : 0.0;
  const z = (world && Number.isFinite(world[2])) ? Number(world[2]) : 0.0;

  const A = normalizeAffine(affine, fallbackSpacing);

  let i = 0, j = 0, k = 0;
  try {
    const inv = matrixInverse4x4(A);
    i = Math.round(inv[0][0] * x + inv[0][1] * y + inv[0][2] * z + inv[0][3]);
    j = Math.round(inv[1][0] * x + inv[1][1] * y + inv[1][2] * z + inv[1][3]);
    k = Math.round(inv[2][0] * x + inv[2][1] * y + inv[2][2] * z + inv[2][3]);
  } catch (_) {
    // Singular matrix fallback: divide by diagonal voxel spacing
    const sx = A[0][0] !== 0 ? Math.abs(A[0][0]) : 1.0;
    const sy = A[1][1] !== 0 ? Math.abs(A[1][1]) : 1.0;
    const sz = A[2][2] !== 0 ? Math.abs(A[2][2]) : 1.0;
    i = Math.round(x / sx);
    j = Math.round(y / sy);
    k = Math.round(z / sz);
  }

  if (bounds && bounds.length >= 3) {
    i = Math.max(0, Math.min(bounds[0] - 1, i));
    j = Math.max(0, Math.min(bounds[1] - 1, j));
    k = Math.max(0, Math.min(bounds[2] - 1, k));
  }

  return [i, j, k];
}

/**
 * Computes voxel spacing along each spatial dimension from affine columns.
 * @param {number[][] | ArrayLike<number> | null} affine - 4x4 matrix
 * @returns {[number, number, number]} [sx, sy, sz] in mm
 */
export function computeVoxelSpacing(affine) {
  const A = normalizeAffine(affine);
  const sx = Math.sqrt(A[0][0] ** 2 + A[1][0] ** 2 + A[2][0] ** 2);
  const sy = Math.sqrt(A[0][1] ** 2 + A[1][1] ** 2 + A[2][1] ** 2);
  const sz = Math.sqrt(A[0][2] ** 2 + A[1][2] ** 2 + A[2][2] ** 2);

  const fx = Number.isFinite(sx) && sx > 0 ? Number(sx.toFixed(3)) : 1.0;
  const fy = Number.isFinite(sy) && sy > 0 ? Number(sy.toFixed(3)) : 1.0;
  const fz = Number.isFinite(sz) && sz > 0 ? Number(sz.toFixed(3)) : 1.0;

  return [fx, fy, fz];
}

/**
 * Computes single voxel volume in mm³ and mL using the determinant of the 3x3 rotation/scale submatrix.
 * @param {number[][] | ArrayLike<number> | null} affine - 4x4 matrix
 * @returns {{ mm3: number, ml: number }}
 */
export function computeVoxelVolume(affine) {
  const A = normalizeAffine(affine);
  const m00 = A[0][0], m01 = A[0][1], m02 = A[0][2];
  const m10 = A[1][0], m11 = A[1][1], m12 = A[1][2];
  const m20 = A[2][0], m21 = A[2][1], m22 = A[2][2];

  const det = m00 * (m11 * m22 - m12 * m21) -
              m01 * (m10 * m22 - m12 * m20) +
              m02 * (m10 * m21 - m11 * m20);

  const absDet = Math.abs(det);
  const mm3 = Number.isFinite(absDet) && absDet > 1e-12 ? absDet : 1.0;
  const ml = mm3 / 1000.0;
  return { mm3: Number(mm3.toFixed(3)), ml: Number(ml.toFixed(6)) };
}

/**
 * Infers 3-letter anatomical orientation code (e.g. 'RAS', 'LPS') from affine direction cosines.
 * @param {number[][] | ArrayLike<number> | null} affine - 4x4 matrix
 * @returns {string} 3-letter orientation string
 */
export function getAnatomicalOrientation(affine) {
  const A = normalizeAffine(affine);
  const labels = [['R', 'L'], ['A', 'P'], ['S', 'I']];
  let orientation = '';

  for (let c = 0; c < 3; c++) {
    let maxVal = -1;
    let maxIdx = 0;
    let isPos = true;

    for (let r = 0; r < 3; r++) {
      const val = A[r][c];
      if (Math.abs(val) > maxVal) {
        maxVal = Math.abs(val);
        maxIdx = r;
        isPos = val >= 0;
      }
    }
    orientation += isPos ? labels[maxIdx][0] : labels[maxIdx][1];
  }

  return orientation;
}
