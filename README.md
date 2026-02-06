# air-canvas-intent-aware
# Intent-Aware Semantic Understanding of Mid-Air Writing and Drawing

This repository implements a unified, sequence-based framework for interpreting mid-air
hand gestures by explicitly inferring user intent (text vs drawing) and enabling
open-vocabulary semantic understanding of air-drawn content.

## Motivation

Most existing air-writing and air-drawing systems treat writing and drawing as separate,
closed-set recognition problems. In real interactions, users naturally switch between
writing text and drawing symbols, expecting the system to infer intent and meaning.

This project explores an intent-aware approach that models mid-air gestures as temporal
sequences, enabling both alignment-free text recognition and open-ended semantic
interpretation of drawings.

## Architecture Overview

The system consists of a shared motion encoder followed by task-specific branches:

- A binary intent inference module (TEXT vs DRAWING)
- A CTC-based text recognition branch
- A contrastive semantic embedding branch for drawings

