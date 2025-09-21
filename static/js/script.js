// Highlight the active nav link while scrolling
const sections = document.querySelectorAll('main section');
const navLinks = document.querySelectorAll('.section-6 a');

window.addEventListener('scroll', () => {
  let current = '';})
function hideAllSections() {
  sections.forEach(section => section.classList.remove('active'));
}

navLinks.forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const targetId = link.getAttribute('href').substring(1);
    const targetSection = document.getElementById(targetId);

    hideAllSections();
    targetSection.classList.add('active');

    navLinks.forEach(nav => nav.classList.remove('active'));
    link.classList.add('active');

    // ✅ If canvas section is activated, resize canvas
    if (targetId === 'canvas') {
      targetSection.classList.add('active');
  // Wait for the section to be displayed, then resize
    setTimeout(resizeCanvas, 0);
}
  });
});
 const canvas = document.getElementById('kolamBoard');
const ctx = canvas.getContext('2d');
const gridInput = document.getElementById('gridSize');
const colorPicker = document.getElementById('colorPicker');

let tool = 'dot';
let drawing = false;
let startX, startY;

// 🔁 Make canvas responsive
function resizeCanvas() {
  const container = canvas.parentElement;
  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;
  drawGrid();
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// 🟩 Draw grid using lines
function drawGrid() {
  const grid = parseInt(gridInput.value);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = '#ddd';
  ctx.lineWidth = 1;

  for (let x = 0; x <= canvas.width; x += grid) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y <= canvas.height; y += grid) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }
}

gridInput.addEventListener('change', drawGrid);
canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mouseup', endDraw);
canvas.addEventListener('mousemove', draw);

function setTool(selected) {
  tool = selected;
}

function startDraw(e) {
  drawing = true;
  startX = e.offsetX;
  startY = e.offsetY;
  if (tool === 'dot') placeDot(e);
}

function endDraw(e) {
  if (tool === 'line' && drawing) {
    ctx.strokeStyle = colorPicker.value;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
  }
  drawing = false;
}

function draw(e) {
  if (!drawing) return;

  if (tool === 'freehand') {
    ctx.fillStyle = colorPicker.value;
    ctx.beginPath();
    ctx.arc(e.offsetX, e.offsetY, 2, 0, Math.PI * 2);
    ctx.fill();
  }
  if (tool === 'eraser') {
    ctx.clearRect(e.offsetX - 5, e.offsetY - 5, 10, 10);
  }
}

function placeDot(e) {
  ctx.fillStyle = colorPicker.value;
  ctx.beginPath();
  ctx.arc(e.offsetX, e.offsetY, 3, 0, Math.PI * 2);
  ctx.fill();
}

function clearBoard() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid();
}

  document.querySelectorAll(".like").forEach(button => {
    button.addEventListener("click", () => {
      button.style.color = "#22c55e"; // green when liked
    });
  });

  document.querySelectorAll(".dislike").forEach(button => {
    button.addEventListener("click", () => {
      button.style.color = "#ef4444"; // red when disliked
    });
  });

  const searchBar = document.getElementById("searchBar");
  const filterCategory = document.getElementById("filterCategory");
  const galleryItems = document.querySelectorAll(".gallery-item");

  function filterGallery() {
    const searchText = searchBar.value.toLowerCase();
    const selectedCategory = filterCategory.value;

    galleryItems.forEach(item => {
      const category = item.getAttribute("data-category");
      const altText = item.querySelector("img").alt.toLowerCase();

      const matchesSearch = altText.includes(searchText);
      const matchesCategory = (selectedCategory === "all" || selectedCategory === category);

      if (matchesSearch && matchesCategory) {
        item.style.display = "inline-block";
      } else {
        item.style.display = "none";
      }
    });
  }

  searchBar.addEventListener("input", filterGallery);
  filterCategory.addEventListener("change", filterGallery);
