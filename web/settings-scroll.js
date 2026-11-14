/* Keep native scrolling; add resistance only after the settings edge is reached. */
(() => {
  const dialog = document.getElementById('settings-dialog');
  if (!dialog) return;
  const content = document.createElement('div');
  content.className = 'settings-content';
  content.append(...dialog.childNodes);
  dialog.append(content);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let pull = 0, releaseTimer, animation, touchY = null;

  function reset(immediate = false) {
    clearTimeout(releaseTimer);
    animation?.cancel();
    const offset = -Math.sign(pull) * 64 * (1 - Math.exp(-Math.abs(pull) / 190));
    pull = 0;
    content.style.transform = '';
    if (!immediate && !reduced.matches && Math.abs(offset) > .5) {
      animation = content.animate([
        {transform: `translateY(${offset}px)`},
        {transform: `translateY(${-offset * .1}px)`, offset: .65},
        {transform: 'translateY(0)'}
      ], {duration: 380, easing: 'cubic-bezier(.2,.8,.3,1)'});
    }
  }
  function hasInnerScroll(target, delta) {
    for (let node = target; node && node !== dialog; node = node.parentElement) {
      const overflow = getComputedStyle(node).overflowY;
      if (!/(auto|scroll)/.test(overflow) || node.scrollHeight <= node.clientHeight + 1) continue;
      if (delta < 0 && node.scrollTop > 0 || delta > 0 && node.scrollTop + node.clientHeight < node.scrollHeight - 1) return true;
    }
    return false;
  }
  function overscroll(delta, event) {
    if (!delta || hasInnerScroll(event.target, delta)) return false;
    const limit = dialog.scrollHeight - dialog.clientHeight;
    const boundary = limit > 0 && (delta < 0 && dialog.scrollTop <= 1 || delta > 0 && dialog.scrollTop >= limit - 1);
    if (!boundary) { if (pull) reset(true); return false; }
    event.preventDefault();
    if (reduced.matches) return true;
    animation?.cancel();
    pull = Math.max(-750, Math.min(750, pull + delta));
    const offset = -Math.sign(pull) * 64 * (1 - Math.exp(-Math.abs(pull) / 190));
    content.style.transform = `translateY(${offset}px)`;
    return true;
  }
  dialog.addEventListener('wheel', event => {
    const delta = event.deltaY * (event.deltaMode === 1 ? 18 : event.deltaMode === 2 ? dialog.clientHeight : 1);
    if (overscroll(delta, event)) {
      clearTimeout(releaseTimer);
      releaseTimer = setTimeout(() => reset(), 110);
    }
  }, {passive: false});
  dialog.addEventListener('touchstart', event => {
    reset(true);
    touchY = event.touches.length === 1 ? event.touches[0].clientY : null;
  }, {passive: true});
  dialog.addEventListener('touchmove', event => {
    if (touchY === null || event.touches.length !== 1) return;
    const next = event.touches[0].clientY;
    overscroll(touchY - next, event);
    touchY = next;
  }, {passive: false});
  ['touchend', 'touchcancel'].forEach(name => dialog.addEventListener(name, () => {touchY = null; reset();}));
  dialog.addEventListener('close', () => reset(true));
})();
