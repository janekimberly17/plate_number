// Same rules as scripts/fleet_check.py, so the website gives the same decisions.

// Characters OCR mixes up on Malaysian plates. Swapping within a group is cheap.
const LOOKALIKE_GROUPS = ['HWMN', '7ZT1', '0QDO', '8B3', '5S', '2Z', '6G', 'VY', '1IL', 'CG']
const LOOKALIKE_COST = 0.3
const PROBABLE_LIMIT = 0.6 // up to two look-alike swaps
const MANUAL_LIMIT = 1.6 // anything further apart is a different plate

function swapCost(a, b) {
  if (a === b) return 0
  if (LOOKALIKE_GROUPS.some((g) => g.includes(a) && g.includes(b))) return LOOKALIKE_COST
  return 1
}

// Edit distance where look-alike swaps cost less than real differences.
export function plateDistance(a, b) {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j)
  for (let i = 1; i <= a.length; i++) {
    const cur = [i]
    for (let j = 1; j <= b.length; j++) {
      cur.push(Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + swapCost(a[i - 1], b[j - 1])))
    }
    prev = cur
  }
  return prev[b.length]
}

// readPlate: plate text without spaces, or null if nothing could be read.
export function verify(readPlate, expectedPlate) {
  if (!readPlate) return { decision: 'MANUAL_CHECK', distance: null }
  const d = plateDistance(readPlate.replaceAll(' ', ''), expectedPlate.replaceAll(' ', '').toUpperCase())
  if (d === 0) return { decision: 'MATCH', distance: d }
  if (d <= PROBABLE_LIMIT + 1e-9) return { decision: 'PROBABLE_MATCH', distance: d }
  if (d <= MANUAL_LIMIT + 1e-9) return { decision: 'MANUAL_CHECK', distance: d }
  return { decision: 'MISMATCH', distance: d }
}
