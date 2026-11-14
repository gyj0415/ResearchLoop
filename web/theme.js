/* Theme colour preference: CSS variables drive every swatch, stored locally.
   Runs before the stylesheet so the remembered palette paints first. */
(() => {
  'use strict';
  const KEY = 'researchloop-theme-color';
  const PALETTES = [
    { id: 'indigo', label: '蓝紫渐变' },
    { id: 'forest', label: '松绿' },
    { id: 'graphite', label: '石墨' },
  ];
  const ids = PALETTES.map(item => item.id);
  const root = document.documentElement;
  let current = 'indigo';
  try {
    const saved = localStorage.getItem(KEY);
    if (ids.includes(saved)) current = saved;
  } catch (_) { /* Storage unavailable: keep the default palette. */ }

  const apply = id => {
    current = ids.includes(id) ? id : 'indigo';
    root.dataset.themeColor = current;
    try { localStorage.setItem(KEY, current); } catch (_) { /* Page still switches. */ }
    updateControls();
  };
  const updateControls = () => {
    document.querySelectorAll('[data-theme-option]').forEach(option => {
      option.setAttribute('aria-checked', String(option.dataset.themeOption === current));
    });
    document.querySelectorAll('meta[name="theme-color"]').forEach(meta => {
      const style = getComputedStyle(document.documentElement);
      const paper = style.getPropertyValue('--paper').trim();
      if (paper) meta.setAttribute('content', paper);
    });
  };
  root.dataset.themeColor = current;
  window.ThemeColor = { palettes: PALETTES, current: () => current, set: apply };
  const init = () => {
    document.querySelectorAll('[data-theme-option]').forEach(option => {
      option.addEventListener('click', () => apply(option.dataset.themeOption));
    });
    updateControls();
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();
