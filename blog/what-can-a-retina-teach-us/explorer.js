(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let pairs, timer = null, showTarget = false, serial = 0;
  const names = () => $('direction').value === 'male_to_female' ? ['Male', 'Female'] : ['Female', 'Male'];
  const current = () => pairs[$('direction').value][Number($('pair').value)];
  const image = url => new Promise((resolve, reject) => { const im = new Image(); im.onload = () => resolve(im); im.onerror = reject; im.src = url; });
  function stopBlink() {
    if (timer !== null) clearInterval(timer);
    timer = null; showTarget = false;
    $('blink-toggle').textContent = 'Start blinking'; $('blink-toggle').setAttribute('aria-pressed', 'false');
    if (pairs) {
      $('blink-image').src = current().images[0];
      $('blink-image').alt = `Pair ${current().index}, synthetic ${names()[0]} source`;
      $('blink-label').textContent = `Source · ${names()[0]}`;
    }
  }
  async function difference(pair, request) {
    const [a, b] = await Promise.all(pair.images.map(image));
    if (request !== serial) return;
    const temp = document.createElement('canvas'); temp.width = temp.height = 512;
    const context = temp.getContext('2d', {willReadFrequently: true});
    context.drawImage(a, 0, 0); const pixelsA = context.getImageData(0, 0, 512, 512);
    context.drawImage(b, 0, 0); const pixelsB = context.getImageData(0, 0, 512, 512);
    const gain = Number($('gain').value), output = context.createImageData(512, 512);
    for (let i = 0; i < output.data.length; i += 4) {
      for (let c = 0; c < 3; c++) output.data[i + c] = Math.min(255, Math.abs(pixelsA.data[i + c] - pixelsB.data[i + c]) * gain);
      output.data[i + 3] = 255;
    }
    $('difference').getContext('2d').putImageData(output, 0, 0);
    $('gain-value').value = `${gain}×`;
  }
  function display() {
    if (!pairs) return;
    stopBlink(); const pair = current(), [source, target] = names(), request = ++serial;
    const sourceIndex = source === 'Female' ? 0 : 1, targetIndex = 1 - sourceIndex;
    const score = (probability) => probability > .999 ? '>99.9%' : `${(100 * probability).toFixed(1)}%`;
    for (const [id, idx] of [['source', 0], ['target', 1], ['wipe-source', 0], ['wipe-target', 1], ['blink-image', 0]]) {
      $(id).src = pair.images[idx]; $(id).alt = `Pair ${pair.index}, synthetic ${idx ? target : source} ${idx ? 'target' : 'source'}`;
    }
    $('source-label').textContent = `Source · CNN ${source.toLowerCase()} ${score(pair.probabilities[0][sourceIndex])}`;
    $('target-label').textContent = `Target · CNN ${target.toLowerCase()} ${score(pair.probabilities[1][targetIndex])}`;
    $('pair-status').textContent = `Pair ${String(pair.index).padStart(2, '0')} of 30 · ${source} → ${target}`;
    $('pair-settings').textContent = `Seed ${pair.seed}. Early-attention threshold ${pair.attention_threshold.toFixed(3)}. Guidance ${pair.guidance.toFixed(3)}. Final EMA diffusion checkpoint, 37,500 updates; CNN checkpoint index 400.`;
    $('previous').disabled = Number($('pair').value) === 0; $('next').disabled = Number($('pair').value) === 29;
    for (const view of ['side', 'wipe', 'blink', 'difference']) $(`${view}-view`).hidden = $('view').value !== view;
    $('blink-label').textContent = `Source · ${source}`;
    if ($('view').value === 'difference') difference(pair, request).catch(() => { $('pair-status').textContent = 'The difference image could not be loaded. Try another pair.'; });
  }
  $('direction').addEventListener('change', () => { $('pair').value = '0'; display(); });
  $('pair').addEventListener('change', display); $('view').addEventListener('change', display);
  $('previous').addEventListener('click', () => { $('pair').value = String(Math.max(0, Number($('pair').value) - 1)); display(); });
  $('next').addEventListener('click', () => { $('pair').value = String(Math.min(29, Number($('pair').value) + 1)); display(); });
  $('reveal').addEventListener('input', () => { $('wipe-target').style.clipPath = `inset(0 ${100 - Number($('reveal').value)}% 0 0)`; });
  $('gain').addEventListener('input', () => { if (pairs) difference(current(), serial); });
  $('blink-toggle').addEventListener('click', () => {
    if (timer !== null) { stopBlink(); $('blink-image').src = current().images[0]; $('blink-label').textContent = `Source · ${names()[0]}`; return; }
    $('blink-toggle').textContent = 'Stop blinking'; $('blink-toggle').setAttribute('aria-pressed', 'true');
    timer = setInterval(() => { showTarget = !showTarget; $('blink-image').src = current().images[Number(showTarget)]; $('blink-label').textContent = `${showTarget ? 'Target' : 'Source'} · ${names()[Number(showTarget)]}`; $('blink-image').alt = `Pair ${current().index}, synthetic ${names()[Number(showTarget)]} ${showTarget ? 'target' : 'source'}`; }, 800);
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stopBlink(); });
  fetch('assets/pairs.json').then(r => { if (!r.ok) throw new Error('Pair data unavailable'); return r.json(); }).then(data => {
    for (const direction of ['male_to_female', 'female_to_male']) if (data[direction].length !== 30) throw new Error('Incomplete pair data');
    pairs = data;
    for (let i = 0; i < 30; i++) { const option = document.createElement('option'); option.value = String(i); option.textContent = `Pair ${String(i + 1).padStart(2, '0')}`; $('pair').append(option); }
    display();
  }).catch(() => { $('pair-status').textContent = 'The pair browser could not load. Open this page through a web server, or view the first pair above.'; });
})();
