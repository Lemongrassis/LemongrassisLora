import { describe, expect, it } from 'vitest'
import { upgradeSnapshot, type GenerateParamsSnapshot } from './paramsSnapshot'

const base: GenerateParamsSnapshot = {
  schema_version: 1, mode: 'xy', model_family: 'krea2', prompts: ['a'], negative_prompt: '',
  width: 1024, height: 1024, steps: 8, cfg_scale: 0, count: 1, seed: 1, loras: [],
  xy_draft: { x: { axis: 'cfg_scale', raw: '0, 3.5', loraIndex: null }, y: null },
}

describe('Krea 2 cfg on ComfyUI scale', () => {
  it('moves old Krea 2 snapshots up by 1, once', () => {
    const up = upgradeSnapshot(base)
    expect(up.cfg_scale).toBe(1)
    expect(up.xy_draft?.x.raw).toBe('1, 4.5')
    expect(up.schema_version).toBe(2)
    expect(upgradeSnapshot(up)).toBe(up)
  })
  it('leaves other families alone', () => {
    const anima = { ...base, model_family: 'anima' as const, cfg_scale: 4 }
    expect(upgradeSnapshot(anima)).toBe(anima)
  })
})
