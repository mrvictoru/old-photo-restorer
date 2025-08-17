// Simple vanilla-JS i18n loader
const DEFAULT_LANG = 'en';
let translations = {};
let currentLang = localStorage.getItem('lang') || DEFAULT_LANG;

async function loadTranslations(lang) {
  try {
    const res = await fetch(`/locales/${lang}.json`);
    translations = await res.json();
  } catch (e) {
    console.warn(`Could not load /locales/${lang}.json`, e);
    translations = {};
  }

  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    const txt = translations[key] || key;
    el.textContent = txt;
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const switcher = document.getElementById('langSwitch');
  switcher.value = currentLang;
  switcher.addEventListener('change', e => {
    currentLang = e.target.value;
    localStorage.setItem('lang', currentLang);
    loadTranslations(currentLang);
  });
  loadTranslations(currentLang);
});
