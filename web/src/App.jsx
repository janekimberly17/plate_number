import { useEffect, useMemo, useState } from 'react'
import { plateDistance, verify } from './verify.js'

const DEMO = `${import.meta.env.BASE_URL}demo`
const GITHUB_URL = 'https://github.com/janekimberly17/yolo_lab_vehicle_plate' 

const DECISIONS = {
  MATCH: { title: 'Approved', text: 'The plate matches the card.' },
  PROBABLE_MATCH: { title: 'Approved', text: 'Only look-alike characters differ (like H and W), so it is treated as the same plate.' },
  MANUAL_CHECK: { title: 'Manual check', text: 'The plate could not be read with certainty. An attendant checks the vehicle.' },
  MISMATCH: { title: 'Held for review', text: 'The plate does not match the vehicle this card was issued to, so the fill is paused for a quick check.' },
}

const STEPS = [
  ['Snapshot', 'The pump takes one photo when the card is inserted.'],
  ['Find the plate', 'YOLOv8 finds the vehicle and its number plate.'],
  ['Read the plate', 'EasyOCR reads it and Malaysian plate rules clean it up.'],
  ['Check the card', 'The plate is compared with the vehicle the card was issued to.'],
]

// Pair up the two plates character by character for the node columns.
function pairChars(read, plate) {
  const a = (read ?? '').split('')
  const b = plate.split('')
  return Array.from({ length: Math.max(a.length, b.length) }, (_, i) => {
    const x = a[i] ?? ''
    const y = b[i] ?? ''
    const status = x === y ? 'same' : x && y && plateDistance(x, y) < 1 ? 'alike' : 'diff'
    return { x, y, status }
  })
}

function Logo() {
  return (
    <svg className="logo" viewBox="0 0 48 32" aria-hidden="true">
      <rect x="2" y="2" width="44" height="28" rx="9" fill="none" stroke="currentColor" strokeWidth="4" />
      <rect x="12" y="11" width="24" height="10" rx="3" fill="currentColor" />
    </svg>
  )
}

function Nodes({ chars, side }) {
  return (
    <div className={`nodes ${side}`}>
      {chars.map((c, i) => (
        <div className="node-row" key={i}>
          <span className="wire" />
          <span className={`node ${c.status}`}>{(side === 'left' ? c.x : c.y) || '·'}</span>
        </div>
      ))}
    </div>
  )
}

export default function App() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [sampleId, setSampleId] = useState(null)
  const [cardId, setCardId] = useState(null)

  useEffect(() => {
    fetch(`${DEMO}/results.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))))
      .then((d) => {
        setData(d)
        selectSample(d, d.samples[0].id)
      })
      .catch(() => setError('Demo results not found. Run scripts/export_demo.py first.'))
  }, [])

  function ownCardOf(d, sample) {
    return d.fleet.find((c) => c.plate === sample.truth)
  }

  function selectSample(d, id) {
    setSampleId(id)
    setCardId(ownCardOf(d, d.samples.find((s) => s.id === id))?.cardId ?? d.fleet[0].cardId)
  }

  const sample = data?.samples.find((s) => s.id === sampleId)
  const card = data?.fleet.find((c) => c.cardId === cardId)
  const ownCard = sample && ownCardOf(data, sample)
  const usingOwnCard = card && card.cardId === ownCard?.cardId
  const result = useMemo(() => (sample && card ? verify(sample.read, card.plate) : null), [sample, card])
  const chars = useMemo(() => (sample && card ? pairChars(sample.read, card.plate) : []), [sample, card])

  function pickOtherCard() {
    const others = data.fleet.filter((c) => c.cardId !== ownCard?.cardId && c.cardId !== cardId)
    setCardId(others[Math.floor(Math.random() * others.length)].cardId)
  }

  return (
    <main className="page">
      <header className="hero">
        <Logo />
        <h1>Fleet Plate Check.</h1>
        <p className="lede">
          Company fuel cards are issued to company vehicles. At the pump, one snapshot confirms the card is being used
          for the vehicle it was issued to. No photos are kept and no personal details are used.
        </p>
      </header>

      {error && <p className="error">{error}</p>}
      {data && sample && (
        <section className="demo" aria-label="Demo">
          <div className="stage">
            <div className="col-label left">Camera read</div>
            <div className="col-label right">Card {card.cardId}</div>
            <Nodes chars={chars} side="left" />
            <img className="photo" src={`${DEMO}/images/${sample.id}.jpg`} alt={`Vehicle with plate ${sample.truth}`} />
            <Nodes chars={chars} side="right" />
          </div>

          {result && (
            <div className={`verdict ${result.decision.toLowerCase()}`} aria-live="polite">
              <strong>{DECISIONS[result.decision].title}</strong>
              <span className="verdict-text">{DECISIONS[result.decision].text}</span>
            </div>
          )}

          <div className="controls">
            <div className="gallery" role="list" aria-label="Choose a vehicle">
              {data.samples.map((s) => (
                <button
                  key={s.id}
                  role="listitem"
                  className={`thumb ${s.id === sampleId ? 'selected' : ''}`}
                  onClick={() => selectSample(data, s.id)}
                  aria-label={`Photo of ${s.truth}`}
                  aria-current={s.id === sampleId}
                >
                  <img src={`${DEMO}/images/${s.id}.jpg`} alt="" loading="lazy" />
                </button>
              ))}
            </div>
            <div className="toggle" role="group" aria-label="Fuel card used">
              <button className={usingOwnCard ? 'on' : ''} onClick={() => setCardId(ownCard?.cardId)}>
                This vehicle's card
              </button>
              <button className={usingOwnCard ? '' : 'on'} onClick={pickOtherCard}>
                Another driver's card
              </button>
            </div>
            <p className="hint">
              Pick a photo, then a card. Teal circles match, amber ones are look-alikes (like H and W), and red ones
              differ.
            </p>
          </div>
        </section>
      )}

      <section className="block">
        <h2>Why this matters in Malaysia.</h2>
        <p>
          Under the subsidised diesel scheme (SKDS 2.0), logistics and public transport operators get fleet cards from
          Shell, Petronas, Petron, Caltex or BHP. Each card is linked to one approved vehicle, and its plate number is
          printed on the card.
        </p>
        <p>
          The misuse is using a card to fill a vehicle it was not issued to. KPDN blocked 223 fleet cards between 2023 and
          May 2026 and is drafting penalties for subsidy abuse. Today a pump attendant is meant to compare the plate on
          the card with the vehicle. This project automates that check.
        </p>
        <p className="muted">A plate check catches the wrong vehicle. It cannot catch diesel pumped into containers for resale.</p>
        <p className="sources">
          <a href="https://www.katsana.com/diesel-subsidy-guide/">How SKDS 2.0 fleet cards work</a>
          <span>·</span>
          <a href="https://www.theborneopost.com/2026/05/16/government-to-tighten-enforcement-of-fleet-card-fuel-subsidy-system-says-kpdn-minister/">
            Fleet card enforcement, Borneo Post
          </a>
        </p>
      </section>

      <section className="block">
        <h2>How it works.</h2>
        <ol className="steps">
          {STEPS.map(([title, text], i) => (
            <li key={title}>
              <span className="step-node">{i + 1}</span>
              <strong>{title}</strong>
              <span className="step-text">{text}</span>
            </li>
          ))}
        </ol>
      </section>

      <footer className="footer">
        <a className="pill" href={GITHUB_URL}>View the code on GitHub</a>
        <p>
          Built with YOLOv8, EasyOCR and React. Photos from the Malaysian Car Plate Dataset,{' '}
          <a href="https://data.mendeley.com/datasets/9795rjwxnd/1">Mendeley Data</a> (CC BY 4.0). Fuel cards and
          drivers are made up. Results come from 89 validation photos, so expect lower accuracy on new photos.
        </p>
      </footer>
    </main>
  )
}
