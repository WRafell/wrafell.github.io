---
permalink: /notes/
layout: home
title: "Notes"
author_profile: false
paginate: true
---

<div style="margin-bottom:2em;">
  <button onclick="toggleAbout()" id="about-btn" style="
    background: transparent;
    border: 1px solid #3a3a3a;
    color: #B8B3AA;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7em;
    letter-spacing: 0.12em;
    padding: 3px 10px;
    cursor: pointer;
    text-transform: uppercase;
    transition: border-color 0.2s, color 0.2s;
  " onmouseover="this.style.borderColor='#9B3D2E';this.style.color='#9B3D2E'" onmouseout="this.style.borderColor='#3a3a3a';this.style.color='#B8B3AA'">About</button>

  <div id="about-panel" style="display:none; margin-top:1.2em; padding:1.2em 0; border-top:1px solid #252525; border-bottom:1px solid #252525;">
    <div style="display:flex; align-items:center; gap:1.6em; margin-bottom:1em;">
      <img src="/assets/images/profile-photo.jpg" alt="Willmer Quinones" style="width:56px; height:56px; object-fit:cover; border-radius:2px; filter:grayscale(20%);">
      <div>
        <div style="font-family:'Cormorant Garamond',serif; font-size:1.2em; font-weight:400; color:#EAE6DC; letter-spacing:0.04em;">Willmer Quinones</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7em; color:#7A756E; letter-spacing:0.06em; text-transform:uppercase; margin-top:2px;">Postdoctoral Fellow · UNIST LAIT</div>
      </div>
    </div>
    <p style="font-size:0.88em; color:#B8B3AA; line-height:1.7; margin:0;">
      AI researcher working on representation learning, multimodal learning, and computational histopathology.
    </p>
  </div>
</div>

<script>
function toggleAbout() {
  var panel = document.getElementById('about-panel');
  var btn = document.getElementById('about-btn');
  if (panel.style.display === 'none') {
    panel.style.display = 'block';
    btn.textContent = 'Close';
  } else {
    panel.style.display = 'none';
    btn.textContent = 'About';
  }
}
</script>
