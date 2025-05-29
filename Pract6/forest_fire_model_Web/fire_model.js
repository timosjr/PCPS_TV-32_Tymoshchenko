const TREE = 0, BURNING = 1, EMPTY = 2;
let grid = [], burnTime = [];
let gridSize = 50, pBurn = 0.6, tBurn = 3;
let cellSize, ctx, interval = null;

function initGrid() {
  grid = Array.from({ length: gridSize }, () => Array(gridSize).fill(TREE));
  burnTime = Array.from({ length: gridSize }, () => Array(gridSize).fill(0));
  let mid = Math.floor(gridSize / 2);
  grid[mid][mid] = BURNING;
  burnTime[mid][mid] = tBurn;
}

function drawGrid() {
  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
      switch (grid[i][j]) {
        case TREE: ctx.fillStyle = "green"; break;
        case BURNING: ctx.fillStyle = "orange"; break;
        case EMPTY: ctx.fillStyle = "red"; break;
      }
      ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
    }
  }
}

function step() {
  let newGrid = grid.map(row => [...row]);
  let newBurnTime = burnTime.map(row => [...row]);

  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
      if (grid[i][j] === TREE) {
        let neighbors = [
          [i - 1, j], [i + 1, j],
          [i, j - 1], [i, j + 1]
        ];
        for (let [ni, nj] of neighbors) {
          if (ni >= 0 && ni < gridSize && nj >= 0 && nj < gridSize) {
            if (grid[ni][nj] === BURNING && Math.random() < pBurn) {
              newGrid[i][j] = BURNING;
              newBurnTime[i][j] = tBurn;
              break;
            }
          }
        }
      } else if (grid[i][j] === BURNING) {
        newBurnTime[i][j]--;
        if (newBurnTime[i][j] <= 0) {
          newGrid[i][j] = EMPTY;
        }
      }
    }
  }

  grid = newGrid;
  burnTime = newBurnTime;
  drawGrid();
}

function startSimulation() {
  stopSimulation(); // зупинити попередню, якщо була
  pBurn = parseFloat(document.getElementById("pBurn").value);
  tBurn = parseInt(document.getElementById("tBurn").value);
  gridSize = parseInt(document.getElementById("gridSize").value);
  const canvas = document.getElementById("forestCanvas");
  canvas.width = canvas.height = 500;
  ctx = canvas.getContext("2d");
  cellSize = canvas.width / gridSize;

  initGrid();
  drawGrid();
  interval = setInterval(step, 200);
}

function stopSimulation() {
  if (interval !== null) {
    clearInterval(interval);
    interval = null;
  }
}
