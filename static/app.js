(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);
  const REF_RATE = 24000;

  const DEFAULTS = { temperature: 0.7, subtemp: 0.6, reppen: 1.0, pause: 250, seed: -1, fixcase: true, numbers: true };
  const SETTING_IDS = Object.keys(DEFAULTS);
  const FORMATTERS = {
    temperature: (v) => Number(v).toFixed(2),
    subtemp: (v) => Number(v).toFixed(2),
    reppen: (v) => Number(v).toFixed(2),
    pause: (v) => `${v} ms`,
  };

  const store = {
    get(key, fallback) {
      try {
        const raw = localStorage.getItem('qtts.' + key);
        return raw === null ? fallback : JSON.parse(raw);
      } catch { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem('qtts.' + key, JSON.stringify(value)); } catch { /* storage unavailable */ }
    },
  };

  async function api(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) {
      let message = res.statusText;
      try { message = (await res.json()).error || message; } catch { /* not json */ }
      throw new Error(message);
    }
    return res;
  }

  let toastTimer = null;
  function toast(message, isError = false) {
    const el = $('#toast');
    el.textContent = message;
    el.className = 'toast' + (isError ? ' error' : '');
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, isError ? 6000 : 3000);
  }

  // ------------------------------------------------------------ settings

  function readSettings() {
    const out = {};
    for (const id of SETTING_IDS) {
      const el = $('#' + id);
      out[id] = el.type === 'checkbox' ? el.checked : Number(el.value);
    }
    return out;
  }

  function applySettings(values) {
    for (const id of SETTING_IDS) {
      const el = $('#' + id);
      if (el.type === 'checkbox') el.checked = !!values[id];
      else el.value = values[id];
      refreshOutput(id);
    }
  }

  function refreshOutput(id) {
    const out = $(`#${id}-val`);
    if (out) out.textContent = FORMATTERS[id]($('#' + id).value);
  }

  function initSettings() {
    applySettings({ ...DEFAULTS, ...store.get('settings', {}) });
    for (const id of SETTING_IDS) {
      $('#' + id).addEventListener('input', () => {
        refreshOutput(id);
        store.set('settings', readSettings());
      });
    }
    $('#resetSettings').addEventListener('click', () => {
      applySettings(DEFAULTS);
      store.set('settings', readSettings());
    });
  }

  // ------------------------------------------------------------ text counter

  function updateCounter() {
    const text = $('#text').value;
    const sentences = text.split(/(?<=[.!?…])\s+|\n+/).filter((s) => s.trim()).length;
    $('#counter').textContent = `${text.length} karakter · ${sentences} cümle`;
    store.set('text', text);
  }

  // ------------------------------------------------------------ audio -> wav

  function encodeWav(samples, rate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const write = (offset, str) => { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); };
    write(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true);
    write(8, 'WAVE'); write(12, 'fmt '); view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    write(36, 'data'); view.setUint32(40, samples.length * 2, true);
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buffer], { type: 'audio/wav' });
  }

  async function toWav(blob) {
    const data = await blob.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    let decoded;
    try { decoded = await ctx.decodeAudioData(data); } finally { ctx.close(); }
    const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * REF_RATE)), REF_RATE);
    const src = offline.createBufferSource();
    src.buffer = decoded;
    src.connect(offline.destination);
    src.start();
    const rendered = await offline.startRendering();
    return { blob: encodeWav(rendered.getChannelData(0), REF_RATE), duration: decoded.duration };
  }

  // ------------------------------------------------------------ voices

  let voices = [];

  async function loadVoices(select) {
    voices = (await (await api('/api/voices')).json()).voices;
    const sel = $('#voice');
    sel.innerHTML = '';
    if (!voices.length) sel.add(new Option('— henüz ses yok: yükleyin veya kaydedin —', ''));
    for (const v of voices) sel.add(new Option(`${v.name}  (${v.duration.toFixed(1)} sn)`, v.name));
    const wanted = select || store.get('voice', '');
    if (voices.some((v) => v.name === wanted)) sel.value = wanted;
    onVoiceChange(false);
  }

  function onVoiceChange(persist = true) {
    const v = voices.find((x) => x.name === $('#voice').value);
    const preview = $('#voicePreview');
    const hint = $('#voiceHint');
    if (v) {
      if (preview.dataset.src !== v.url) { preview.src = v.url; preview.dataset.src = v.url; }
      preview.hidden = false;
      const tooShort = v.duration < 3, tooLong = v.duration > 12;
      hint.hidden = !(tooShort || tooLong);
      hint.textContent = tooShort ? 'Referans çok kısa. 5–10 sn en iyisidir.' : tooLong ? `Referans uzun (${v.duration.toFixed(0)} sn). 5–10 sn en iyisidir.` : '';
      if (persist) store.set('voice', v.name);
    } else {
      preview.removeAttribute('src');
      delete preview.dataset.src;
      preview.hidden = true;
      hint.hidden = true;
    }
    $('#deleteVoice').disabled = !v;
  }

  let pending = null;

  async function stagePending(blob, suggestedName) {
    try {
      const wav = await toWav(blob);
      if (pending) URL.revokeObjectURL(pending.url);
      pending = { blob: wav.blob, url: URL.createObjectURL(wav.blob) };
      $('#pendingAudio').src = pending.url;
      $('#pendingInfo').textContent = `· ${wav.duration.toFixed(1)} sn`;
      $('#pendingName').value = suggestedName;
      $('#pendingBox').hidden = false;
      $('#pendingName').focus();
      $('#pendingName').select();
    } catch (err) {
      toast('Ses çözülemedi: ' + err.message, true);
    }
  }

  function clearPending() {
    if (pending) URL.revokeObjectURL(pending.url);
    pending = null;
    $('#pendingAudio').removeAttribute('src');
    $('#pendingBox').hidden = true;
  }

  async function savePending() {
    if (!pending) return;
    const name = $('#pendingName').value.trim();
    if (!name) { toast('Sese bir ad verin.', true); return; }
    try {
      const res = await api('/api/voices?name=' + encodeURIComponent(name), { method: 'POST', body: pending.blob });
      const saved = (await res.json()).name;
      clearPending();
      await loadVoices(saved);
      toast(`“${saved}” sesi kaydedildi`);
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function deleteVoice() {
    const name = $('#voice').value;
    if (!name || !confirm(`“${name}” sesi silinsin mi?`)) return;
    try {
      await api('/api/voices/' + encodeURIComponent(name), { method: 'DELETE' });
      await loadVoices();
    } catch (err) {
      toast(err.message, true);
    }
  }

  let recorder = null;

  async function toggleRecord() {
    const btn = $('#recBtn');
    if (recorder) { recorder.stop(); return; }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      toast('Mikrofon kullanılamıyor: ' + err.message, true);
      return;
    }
    const chunks = [];
    const mr = new MediaRecorder(stream);
    const t0 = Date.now();
    const timer = setInterval(() => { btn.textContent = `■ Durdur  ${Math.round((Date.now() - t0) / 1000)} sn`; }, 250);
    mr.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    mr.onstop = async () => {
      clearInterval(timer);
      stream.getTracks().forEach((t) => t.stop());
      recorder = null;
      btn.textContent = '● Kaydet';
      btn.classList.remove('recording');
      await stagePending(new Blob(chunks, { type: mr.mimeType }), 'recording');
    };
    recorder = mr;
    btn.textContent = '■ Durdur  0 sn';
    btn.classList.add('recording');
    mr.start();
  }

  // ------------------------------------------------------------ generation

  let busy = false;
  let pollTimer = null;

  function setBusy(value) {
    busy = value;
    $('#generate').disabled = value;
    $('#stop').hidden = !value;
    $('#progressWrap').hidden = !value;
    $('#progressText').textContent = '';
    if (value) {
      $('#progressWrap').classList.add('indeterminate');
      $('#progressBar').style.width = '0';
    }
  }

  async function pollProgress() {
    try {
      const s = await (await api('/api/status')).json();
      const { done, total } = s.progress;
      if (total) {
        $('#progressWrap').classList.remove('indeterminate');
        $('#progressBar').style.width = `${(done / total) * 100}%`;
        $('#progressText').textContent = `${done}/${total}`;
      }
    } catch { /* ignore transient errors */ }
  }

  async function generate() {
    if (busy) return;
    if (!modelReady) { toast('Model hâlâ yükleniyor, birkaç saniye bekleyin.', true); return; }
    const text = $('#text').value.trim();
    if (!text) { toast('Önce bir metin girin.', true); return; }
    const voice = $('#voice').value;
    if (!voice) { toast('Önce bir referans ses ekleyin veya seçin.', true); return; }

    setBusy(true);
    pollTimer = setInterval(pollProgress, 400);
    try {
      const res = await api('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, voice, ...readSettings() }),
      });
      const r = await res.json();
      const player = $('#player');
      player.src = r.url;
      $('#download').href = r.url;
      $('#download').setAttribute('download', r.file);
      $('#resultMeta').textContent = `${r.duration.toFixed(1)} sn ses · ${r.elapsed.toFixed(1)} sn · RTF ${r.rtf} · ${r.chunks} parça`;
      $('#resultCard').hidden = false;
      player.play().catch(() => { /* autoplay blocked */ });
      loadHistory();
    } catch (err) {
      toast(err.message, err.message !== 'İptal edildi.');
    } finally {
      clearInterval(pollTimer);
      setBusy(false);
    }
  }

  async function stopGeneration() {
    try { await api('/api/cancel', { method: 'POST' }); } catch { /* ignore */ }
  }

  // ------------------------------------------------------------ history

  function formatTime(epoch) {
    return new Date(epoch * 1000).toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
  }

  async function loadHistory() {
    let items;
    try { items = (await (await api('/api/outputs')).json()).outputs; } catch { return; }
    const box = $('#history');
    box.textContent = '';
    if (!items.length) {
      const p = document.createElement('p');
      p.className = 'muted small';
      p.textContent = 'Henüz bir şey üretilmedi.';
      box.append(p);
      return;
    }
    for (const it of items) {
      const el = document.createElement('div');
      el.className = 'hist';

      const text = document.createElement('div');
      text.className = 'hist-text';
      text.textContent = it.text || it.file;
      text.title = it.text || '';

      const meta = document.createElement('div');
      meta.className = 'hist-meta';
      meta.textContent = `${it.voice || '—'} · ${Number(it.duration).toFixed(1)} sn · ${formatTime(it.created)}`;

      const audio = document.createElement('audio');
      audio.controls = true;
      audio.preload = 'none';
      audio.src = it.url;

      const row = document.createElement('div');
      row.className = 'row';

      const reuse = document.createElement('button');
      reuse.className = 'btn small';
      reuse.textContent = 'Metni kullan';
      reuse.addEventListener('click', () => {
        $('#text').value = it.text || '';
        updateCounter();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });

      const dl = document.createElement('a');
      dl.className = 'btn small';
      dl.textContent = 'İndir';
      dl.href = it.url;
      dl.download = it.file;

      const del = document.createElement('button');
      del.className = 'btn small danger';
      del.textContent = 'Sil';
      del.addEventListener('click', async () => {
        try {
          await api('/api/outputs/' + encodeURIComponent(it.file), { method: 'DELETE' });
          loadHistory();
        } catch (err) { toast(err.message, true); }
      });

      row.append(reuse, dl, del);
      el.append(text, meta, audio, row);
      box.append(el);
    }
  }

  // ------------------------------------------------------------ status

  let modelReady = false;

  async function loadStatus() {
    const pill = $('#status');
    try {
      const s = await (await api('/api/status')).json();
      if (s.error) {
        pill.textContent = 'hata: ' + s.error;
        pill.className = 'pill err';
        return;
      }
      if (!s.ready) {
        pill.textContent = s.stage || 'hazırlanıyor…';
        pill.className = 'pill';
        setTimeout(loadStatus, 1000);
        return;
      }
      modelReady = true;
      pill.textContent = `${s.model_id} · ${s.device} · ${s.dtype}`;
      pill.className = 'pill ok';
    } catch {
      pill.textContent = 'sunucuya ulaşılamıyor';
      pill.className = 'pill err';
    }
  }

  // ------------------------------------------------------------ init

  function init() {
    initSettings();
    $('#text').value = store.get('text', '');
    updateCounter();
    $('#text').addEventListener('input', updateCounter);
    $('#text').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate(); }
    });

    $('#generate').addEventListener('click', generate);
    $('#stop').addEventListener('click', stopGeneration);
    $('#voice').addEventListener('change', () => onVoiceChange(true));
    $('#deleteVoice').addEventListener('click', deleteVoice);
    $('#recBtn').addEventListener('click', toggleRecord);
    $('#pendingSave').addEventListener('click', savePending);
    $('#pendingDiscard').addEventListener('click', clearPending);
    $('#pendingName').addEventListener('keydown', (e) => { if (e.key === 'Enter') savePending(); });
    $('#refreshHistory').addEventListener('click', loadHistory);
    $('#voiceFile').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      e.target.value = '';
      if (file) await stagePending(file, file.name.replace(/\.[^.]+$/, ''));
    });

    loadStatus();
    loadVoices().catch((err) => toast(err.message, true));
    loadHistory();
  }

  init();
})();
