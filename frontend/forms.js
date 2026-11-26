const form = document.querySelector('form');
const result = document.querySelector('#result');
function show(text, error = false) {
  result.hidden = false;
  result.dataset.error = String(error);
  result.textContent = text;
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  const fields = Array.from(form.elements);
  fields.forEach(field => field.disabled = true);
  try {
    const text = document.querySelector('#prompt').value;
    if (!text.trim()) return show('Enter a prompt first.', true);
    show('Classifying…');
    const response = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
    if (!response.ok) throw new Error(response.status === 422 ? 'Check the prompt length and try again.' : await response.text());
    const p = await response.json();
    show((p.label === 'attack' ? 'Potential injection' : 'Safe') + ' · ' + (100 * p.confidence).toFixed(1) + '% model confidence' + (p.truncated ? '\nOnly the first 192 tokens were evaluated.' : ''));
  } catch (error) { show(error.message || 'Request failed.', true); }
  finally { fields.forEach(field => field.disabled = false); }
});
