# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 00:01:27 2024
I had a thought... why not show what we are doing with a simulation.

@author: apod
"""

let waves = [];
let frame;
let framePos;
let skyColor;
let cloudColor;
let sunColor;

function setup() {
  createCanvas(800, 600);
  
  // Initialize waves
  for (let i = 0; i < 7; i++) {
    waves.push({
      frequency: random(0.01, 0.03),
      speed: random(1, 2),
      amplitude: 30,
      color: color(0, 100 + random(50), 200 + random(55))
    });
  }
  
  // Initialize frame
  frame = {
    x: 0,
    y: height - 50,
    size: 30,
    currentWave: 0
  };
  
  framePos = createVector(frame.x, frame.y);
  
  // Set colors
  skyColor = color(255, 200, 100);
  cloudColor = color(255, 230, 180);
  sunColor = color(255, 200, 0);
}

function draw() {
  background(220);
  
  // Draw sky
  fill(skyColor);
  rect(0, 0, width, height/3);
  
  // Draw clouds
  fill(cloudColor);
  ellipse(100, 50, 80, 50);
  ellipse(300, 30, 100, 60);
  ellipse(600, 70, 120, 70);
  
  // Draw sun
  fill(sunColor);
  arc(width - 100, height/3, 100, 100, PI, TWO_PI);
  
  // Draw and animate waves
  for (let i = 0; i < waves.length; i++) {
    let wave = waves[i];
    stroke(wave.color);
    strokeWeight(3);
    noFill();
    
    beginShape();
    for (let x = 0; x < width; x++) {
      let y = height - 50 - i * 40 + sin(x * wave.frequency + frameCount * 0.05) * wave.amplitude;
      vertex(x, y);
      
      # // Update frame position if it's on this wave
      if (i === frame.currentWave) {
        framePos.y = y;
      }
    }
    endShape();
    
    // Move wave
    wave.frequency += 0.0001 * wave.speed;
  }
  
  // Draw and animate frame
  fill(255);
  ellipse(framePos.x, framePos.y, frame.size);
  
  // Move frame
  framePos.x += waves[frame.currentWave].speed;
  
  # // Jump to next wave if it's higher
  if (frame.currentWave < waves.length - 1) {
    let nextWaveY = height - 50 - (frame.currentWave + 1) * 40 + 
                    sin(framePos.x * waves[frame.currentWave + 1].frequency) * waves[frame.currentWave + 1].amplitude;
    if (nextWaveY < framePos.y) {
      frame.currentWave++;
    }
  }
  
  // Final animation
  if (framePos.x > width) {
    noLoop();
    // Here you would implement the frame flying to center and enlarging
  }
}