const form = document.querySelector('form');
const result = document.querySelector('#result');
const evidence = document.querySelector('#evidence');
form.addEventListener('input', () => {
  result.hidden = true;
  evidence.hidden = true;
});
function show(text, error = false) {
  result.hidden = false;
  result.dataset.error = String(error);
  result.textContent = text;
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  evidence.hidden = true;
  const fields = Array.from(form.elements);
  fields.forEach(field => field.disabled = true);
  try {
    const text = document.querySelector('#prompt').value;
    if (!text.trim()) return show('Enter a prompt first.', true);
    show('Scanning document…');
    const response = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
    if (!response.ok) throw new Error(response.status === 422 ? 'Check the prompt length and try again.' : await response.text());
    const p = await response.json();
    const peak = p.highest_scoring_passage;
    if (!peak) throw new Error('Restart the updated classifier API to scan full documents.');
    const verdict = p.label === 'attack' ? 'Potential injection' : 'No injection flagged';
    const unit = p.passages_checked === 1 ? 'passage' : 'passages';
    show(verdict + '\n' + p.passages_checked + ' of ' + p.passages_checked + ' ' + unit + ' checked.');
    document.querySelector('#location').textContent = 'Passage ' + peak.number + ' · Injection score ' + p.attack_score.toFixed(3);
    // API offsets count Unicode code points. Array.from also counts emoji as one code point.
    const characters = Array.from(text);
    const mark = document.createElement('mark');
    mark.textContent = characters.slice(peak.start, peak.end).join('');
    const prefix = characters.slice(Math.max(0, peak.start - 60), peak.start).join('');
    const suffix = characters.slice(peak.end, peak.end + 60).join('');
    document.querySelector('#excerpt').replaceChildren(
      (peak.start > 60 ? '…' : '') + prefix, mark,
      suffix + (peak.end + 60 < characters.length ? '…' : '')
    );
    evidence.hidden = false;
  } catch (error) { show(error.message || 'Request failed.', true); }
  finally { fields.forEach(field => field.disabled = false); }
});
