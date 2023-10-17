const m=30; 
const n=70; 
const scenes=50;
const c = [54., 40, 53, 73, 77, 77, 50, 50, 73, 73, 80, 58, 67, 79, 71, 46, 43, 42, 42, 63, 79, 43, 51, 41, 41, 42, 56, 50, 65, 77]
const d = [19. 5 17 5 16 17 2 8 16 6 15 9 9 11 25 12 11 8 21 11 13 9 7 15 5 15 9 1 2 14; 24 22 16 10 8 1 11 3 17 10 3 11 19 11 14 21 14 2 12 5 9 10 4 5 17 14 8 7 0 24; 18 0 18 8 23 5 25 3 16 24 10 22 12 1 15 10 5 23 5 1 17 9 7 22 25 24 1 1 22 12; 7 17 20 1 4 11 7 13 22 17 14 16 10 4 14 18 2 9 13 4 13 8 19 5 3 13 6 19 9 2; 14 3 22 22 1 19 17 5 4 1 22 10 15 10 13 15 4 20 20 18 3 20 5 11 10 15 7 25 15 1; 0 25 18 25 17 25 24 8 6 22 11 15 7 12 12 7 25 0 9 6 23 23 6 7 9 0 0 10 9 20; 1 18 5 16 16 16 7 12 25 21 15 17 2 20 11 12 20 13 9 6 18 22 12 18 15 24 10 21 22 22; 12 13 10 5 18 0 10 0 11 17 9 10 0 24 3 0 24 19 21 11 3 11 7 20 1 2 6 9 23 23; 16 11 19 4 4 7 24 12 23 9 24 14 8 16 7 9 20 25 4 16 24 16 4 9 17 0 17 11 15 1; 12 13 22 3 16 19 4 20 23 8 16 19 18 12 25 4 25 23 13 20 12 8 24 18 24 25 20 9 16 13; 23 23 13 8 23 10 2 7 19 4 20 5 15 12 0 10 3 20 15 2 11 15 17 23 7 20 17 14 5 5; 17 18 20 18 16 24 13 1 25 1 15 15 5 21 7 11 13 1 7 15 5 18 3 9 14 16 22 16 1 13; 20 0 23 12 22 5 5 22 8 0 13 22 21 18 12 11 6 0 25 9 13 2 5 3 13 10 1 11 9 6; 14 9 1 17 4 4 4 19 3 25 8 7 3 1 20 4 18 15 25 4 13 5 11 18 16 8 5 8 22 3; 5 1 1 20 11 14 14 9 21 15 5 6 0 5 6 8 12 11 9 17 4 1 14 14 15 8 13 21 9 10; 24 9 1 24 25 0 1 13 3 9 5 9 12 0 21 13 12 20 2 10 5 22 11 14 24 24 2 1 11 25; 3 4 9 2 8 13 17 24 4 10 15 19 11 12 10 0 9 9 18 6 18 25 25 3 17 1 15 3 10 21; 15 4 16 1 8 25 16 20 9 19 20 9 9 1 1 18 13 18 22 17 6 0 15 16 7 19 24 16 24 19; 12 5 23 21 21 5 9 2 8 20 25 5 7 20 1 0 7 13 13 7 12 5 23 5 15 13 21 23 12 5; 18 16 18 17 11 12 4 23 24 5 5 8 11 1 14 1 25 12 25 17 11 18 5 14 10 12 5 7 1 23; 1 15 18 9 17 25 6 2 19 16 14 5 15 13 7 7 24 17 6 7 2 4 6 5 2 8 15 25 3 17; 19 9 8 5 18 25 1 4 0 18 18 22 12 8 9 24 17 1 19 15 15 2 13 13 5 22 4 18 23 15; 6 18 2 12 6 10 22 2 10 15 24 11 17 23 7 4 0 12 12 10 19 23 9 25 13 23 10 8 11 20; 9 12 5 15 18 15 13 7 1 24 22 11 15 10 21 12 2 18 23 2 4 16 2 18 10 17 2 15 5 14; 16 8 1 18 3 3 4 17 13 1 7 3 20 20 0 4 16 11 19 17 1 18 12 9 9 10 22 6 22 9; 17 24 24 20 11 11 0 21 5 8 5 13 0 3 10 19 8 5 18 7 5 7 23 14 15 0 2 9 23 10; 25 6 18 3 11 25 19 10 12 17 8 14 1 3 13 15 23 14 0 9 23 19 14 18 8 18 1 2 24 10; 11 4 17 20 11 20 12 18 6 12 18 20 16 5 14 8 2 2 2 18 9 23 22 16 2 2 10 6 14 16; 4 19 20 3 24 17 15 4 25 3 7 1 20 18 25 10 8 20 22 10 7 10 7 25 24 9 23 7 13 0; 9 22 12 16 22 17 11 9 20 17 21 23 22 21 1 18 6 12 6 23 22 13 18 1 8 2 25 7 12 5; 14 1 22 10 14 13 10 3 15 25 12 15 1 13 3 25 7 9 4 13 23 13 4 5 9 17 7 20 10 13; 1 24 16 15 12 19 2 16 6 5 25 23 1 7 11 1 19 5 15 3 6 23 8 9 2 12 10 7 19 1; 19 23 6 3 15 25 17 11 15 13 4 10 12 12 9 23 9 6 8 0 5 1 15 25 2 14 9 22 11 22; 9 1 7 16 20 19 5 16 24 11 2 9 25 18 9 12 17 21 23 8 20 25 12 25 10 14 2 14 22 3; 22 5 22 15 5 0 18 24 19 24 11 21 11 21 20 4 6 5 8 8 7 6 23 20 19 23 21 15 14 18; 1 16 3 5 10 3 21 8 5 18 3 11 9 5 5 4 17 16 25 25 10 0 21 14 5 25 7 8 7 24; 25 23 2 14 18 11 23 5 4 20 22 14 1 3 21 20 22 13 23 0 24 15 1 15 24 5 14 22 1 22; 12 5 9 22 22 19 21 14 19 3 19 19 24 18 8 20 1 11 24 7 11 18 15 21 11 11 4 6 22 13; 16 12 13 13 10 25 1 18 14 9 11 8 9 24 2 5 19 25 2 0 20 20 20 13 0 5 3 19 25 25; 8 3 11 1 20 7 2 0 7 1 11 6 10 3 13 15 8 20 10 6 25 8 12 14 13 12 1 19 3 9; 15 21 7 2 2 4 5 21 19 1 10 22 18 18 17 19 13 11 20 5 7 12 16 4 25 9 15 1 13 25; 12 13 11 3 17 19 13 25 22 13 6 11 6 13 6 20 17 20 5 3 12 11 19 7 2 22 13 10 23 21; 2 16 7 2 16 11 0 16 14 13 19 21 17 0 23 0 7 1 5 18 15 13 12 20 8 22 17 18 21 12; 8 14 21 14 5 10 25 25 1 20 10 11 17 2 14 11 18 7 11 16 18 17 3 23 11 10 15 4 25 13; 25 0 18 14 16 21 25 1 23 2 15 16 9 0 1 4 23 4 14 23 20 14 12 16 22 8 18 1 13 4; 22 25 22 5 20 3 1 16 18 1 13 13 19 6 17 14 4 8 4 20 20 14 21 10 3 16 17 4 18 24; 15 14 14 20 12 21 22 22 17 19 20 0 19 0 21 24 18 1 23 18 13 8 20 8 17 16 4 11 25 18; 0 4 18 7 24 22 0 4 14 23 10 7 12 0 5 22 24 18 17 10 5 12 13 21 4 4 20 5 21 2; 14 18 0 3 3 8 11 5 3 9 13 9 1 5 21 23 5 21 5 21 2 11 14 12 19 12 4 15 5 17; 7 22 14 0 3 0 13 19 15 25 21 13 22 20 10 19 1 17 10 5 17 11 23 21 13 18 19 6 20 22; 24 19 2 19 21 22 7 13 1 0 21 13 6 15 12 20 1 7 14 21 21 5 21 19 6 11 14 8 25 0; 15 5 11 2 21 20 5 16 19 14 4 17 9 13 15 14 25 16 11 9 21 15 24 4 16 2 7 1 19 3; 14 7 19 3 1 5 8 0 17 12 24 6 0 3 13 22 19 25 15 14 0 0 6 7 13 23 19 4 23 25; 16 19 3 20 1 7 23 4 18 21 15 24 9 12 9 24 7 24 10 7 18 3 5 13 16 24 20 7 24 13; 3 12 12 12 6 19 20 3 0 25 17 24 7 13 7 7 19 3 21 18 10 20 16 16 4 6 25 1 17 12; 1 15 3 6 4 21 13 6 22 3 9 18 0 19 17 25 14 0 6 2 23 10 21 10 8 24 11 2 4 1; 2 7 19 13 24 14 25 20 15 0 11 15 1 14 12 1 24 21 25 13 4 10 3 0 8 14 17 3 15 1; 7 16 24 6 20 10 15 21 4 15 12 21 7 4 1 24 9 11 23 21 0 2 12 23 7 2 11 12 16 9; 17 6 1 13 25 2 8 14 23 4 4 8 16 22 15 16 10 10 16 9 18 15 7 4 5 6 1 13 4 3; 19 4 5 12 3 3 8 22 17 18 12 13 9 23 17 13 11 8 21 24 24 1 7 15 21 22 16 22 11 15; 7 24 24 0 16 18 4 13 15 9 11 8 15 19 17 25 5 12 2 18 22 9 13 0 7 6 18 9 16 21; 13 23 23 7 20 4 25 4 0 18 13 25 4 1 23 4 4 18 8 15 15 5 3 17 23 20 25 10 2 0; 2 3 17 15 6 2 19 1 0 18 23 5 23 8 0 18 9 2 23 4 4 7 0 3 9 19 18 17 8 20; 15 21 14 1 9 2 3 24 13 15 13 5 20 9 13 4 25 7 4 5 9 15 9 17 18 7 11 25 16 23; 25 3 15 1 2 6 4 16 24 6 12 15 25 18 16 18 8 21 0 12 23 10 15 13 25 6 6 23 4 16; 15 24 13 5 17 14 24 11 8 24 0 15 4 5 13 11 22 2 6 24 14 17 10 21 2 14 19 22 15 7; 22 12 1 1 3 24 10 0 20 2 7 11 3 13 18 13 1 13 5 17 19 25 17 8 22 13 12 11 19 11; 8 0 6 7 0 3 17 17 0 12 0 13 18 3 25 17 23 6 24 9 14 16 5 1 20 12 1 16 18 15; 3 4 24 20 1 7 18 25 3 2 16 8 9 2 22 25 4 21 4 20 1 9 22 21 1 18 18 20 2 0; 6 24 14 24 10 13 10 5 16 0 10 3 16 9 0 21 17 4 22 13 5 11 19 15 8 5 22 13 6 25]
const q0 = fill(1e3,m)
const u = 1291.15
const h = [0. 0 0 1 1 1 1 1 0 1 0 0 0 0 1 0 1 0 0 1 1 1 1 0 1 1 0 1 1 1 0 0 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 1 1 0; 0 0 0 0 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 1 1 0 0 0 0 0 0 0 1 0 1 1 1 0 1 1 0 0 1 0 1 1 0 1 1 0 0 1 0 0; 1 1 0 0 1 1 0 1 1 1 0 0 1 1 0 1 0 0 0 0 0 1 1 0 1 0 1 1 1 1 1 0 1 1 0 0 1 1 0 0 1 1 1 0 0 1 1 1 1 0; 0 1 1 1 1 0 0 1 1 0 1 1 0 0 1 0 1 0 1 1 0 1 0 1 0 0 0 1 0 1 1 1 0 1 0 0 0 0 0 0 1 0 1 1 1 0 0 1 0 1; 1 0 0 0 1 1 0 1 1 1 0 1 1 0 1 1 0 1 1 0 0 1 1 0 0 0 1 1 0 0 0 1 0 0 0 0 0 1 0 1 0 1 1 0 0 1 1 0 0 1; 0 0 0 1 1 0 1 0 0 0 1 1 1 0 1 1 1 0 1 1 1 1 1 0 0 1 0 1 0 0 0 0 0 1 0 1 1 0 1 0 1 1 0 0 0 0 0 1 1 1; 0 0 1 1 1 0 0 1 1 1 0 1 0 1 1 0 0 1 0 0 1 1 1 1 0 0 0 1 0 1 0 1 1 1 1 0 1 1 0 1 0 1 1 0 0 0 1 1 0 1; 1 1 1 1 1 1 1 0 0 0 1 1 1 0 0 0 0 0 0 0 1 1 0 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0; 0 1 1 1 0 0 0 0 0 1 0 0 1 0 1 0 1 1 0 1 0 1 0 0 0 0 1 1 1 0 1 1 1 0 1 0 1 0 0 1 1 0 0 1 1 0 0 0 1 0; 0 1 1 1 1 1 0 1 0 0 0 0 1 1 0 1 1 0 1 1 0 0 1 0 1 0 0 0 1 0 1 1 0 0 1 0 1 0 0 1 1 0 0 0 1 1 0 1 0 0; 0 1 0 0 0 1 0 1 1 1 1 0 1 1 1 1 0 0 1 0 1 0 0 1 1 0 0 1 1 0 1 0 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 1; 1 0 0 1 1 1 1 0 0 1 1 0 1 0 0 1 1 1 1 0 0 1 1 1 0 1 1 1 0 1 0 0 1 0 1 0 1 0 1 0 0 0 0 0 1 0 0 0 0 1; 1 1 1 1 1 0 0 1 0 0 1 0 0 1 0 0 1 0 1 0 0 0 0 1 1 1 0 0 0 1 1 0 1 0 1 0 1 1 1 0 1 0 1 1 1 1 0 0 1 0; 0 1 1 1 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 1 0 1 0 0 1 1 1 0 1 1 0 1 1 1 1 1 0 1 0 1 0 0 1 0 1 0 1 0 0 0; 0 1 0 0 1 0 0 1 1 0 0 1 0 1 0 1 0 1 1 1 0 1 1 1 1 1 0 0 1 0 0 0 0 1 1 0 1 1 1 0 1 1 0 0 0 0 1 0 0 1; 1 0 0 0 0 1 1 0 1 0 1 0 1 1 1 1 0 0 1 0 0 1 0 0 1 0 0 1 1 1 1 1 0 0 0 1 1 0 1 0 1 1 1 0 0 0 0 1 1 0; 1 1 1 0 1 0 0 0 0 1 1 0 1 0 1 1 1 0 0 0 0 1 1 1 0 0 1 0 0 0 1 0 0 0 1 0 1 0 0 1 0 0 1 1 0 0 0 0 1 0; 1 0 1 0 1 0 1 0 1 1 1 0 1 0 0 0 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 0 1 0 1 1 0 0 1 0 0 0 1 1 0 0 0 0 0 1; 0 0 0 1 0 0 0 1 1 1 0 1 0 0 1 1 1 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 1 0 1 1 0 1 1 0 1 0 0 0 0 0 1 0 1; 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 1 1 1 0 1 0 1 1 0 1 0 0 1 1 0 0 0 1 0 0 0 1 1 0 1 0 1 1 1 0 0 0 0 1; 1 0 1 1 0 1 0 1 1 0 1 0 1 1 1 1 0 0 1 0 1 1 1 1 1 1 0 0 1 1 0 0 1 0 0 0 1 0 0 1 0 0 0 0 1 1 1 0 0 0; 0 1 1 0 0 0 0 1 1 1 0 0 1 0 0 1 0 1 1 0 0 0 1 1 0 1 1 1 1 1 0 0 0 0 1 0 0 1 0 0 0 0 1 0 0 0 1 1 1 1; 0 0 0 1 1 0 0 1 0 1 1 1 1 0 0 0 1 0 0 0 0 0 1 1 1 1 0 0 0 1 0 0 1 0 1 0 1 0 1 0 0 1 1 0 1 1 0 0 0 0; 0 0 0 1 0 0 0 0 0 1 0 1 1 0 1 1 0 0 1 0 1 1 1 1 0 1 1 1 1 0 0 1 0 0 0 1 0 1 1 1 1 1 0 0 0 0 0 0 0 0; 0 0 0 0 1 1 0 1 0 0 1 0 0 0 1 0 1 0 1 1 1 0 1 1 1 1 0 0 1 1 1 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 1 1 0; 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 1 1 0 0 1 0 1 1 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1 1 0 0 0 1 1 0; 0 1 0 0 0 0 1 0 1 1 0 1 1 0 0 0 1 0 1 1 1 1 1 0 1 1 1 0 0 0 1 1 0 0 0 1 0 0 0 1 1 1 1 1 1 0 1 0 0 0; 1 0 0 0 0 1 0 1 0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 0 0 0 0 1 1 1 0 0 1 0 1 1 0 0 1 0 0 1 1 1 0 1 1 0 0 0; 0 0 0 1 0 0 1 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 0 1 1 1 0 1 1 1 1 0 0 1 1 1 1 1 0 1 0 0 0 1 0 1 0 1; 0 1 1 0 0 0 0 0 1 0 1 1 1 0 1 0 0 1 0 0 0 1 1 0 0 0 0 1 1 1 1 1 0 1 0 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0; 1 1 1 1 1 0 0 1 0 1 0 0 1 0 1 0 1 0 1 0 1 1 1 1 1 1 0 1 1 0 0 1 0 0 1 0 1 1 0 1 1 1 0 1 0 0 1 0 1 0; 0 1 1 0 0 0 1 0 0 0 0 0 1 1 0 1 1 1 1 1 0 1 1 0 1 0 1 1 0 0 1 0 0 1 1 1 1 0 0 1 0 0 1 0 1 0 0 0 1 1; 0 1 1 0 0 1 0 0 0 0 1 1 1 1 1 0 1 1 1 1 0 1 1 0 0 1 0 1 0 0 0 0 1 0 1 1 0 0 0 0 0 1 1 0 0 0 1 1 1 0; 1 1 0 0 1 0 1 0 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 1 1 1 0 0 1 1 0 0 0 0 1 1 1 1 0 0 1 0 1 1 0 0 0 1 1 0; 0 0 1 1 0 1 1 1 1 0 1 0 1 0 0 1 0 0 1 1 1 1 0 1 0 0 1 0 0 0 0 0 1 0 0 1 1 0 1 1 1 0 0 1 1 0 0 0 0 0; 0 1 1 1 1 0 0 0 1 1 0 1 1 0 1 1 0 1 0 1 0 0 0 0 1 1 0 1 1 1 1 1 1 1 1 0 0 0 0 1 1 0 0 1 1 1 1 0 1 0; 1 1 0 0 0 1 0 0 1 1 0 0 0 0 0 1 1 0 0 1 0 0 0 1 1 1 1 0 1 1 0 0 0 1 1 1 1 0 0 1 0 1 1 1 0 1 0 1 0 0; 0 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 1 1 1 1 0 1 0 1 0 0 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0; 0 1 1 0 1 1 0 0 0 1 1 1 0 1 0 1 1 1 0 0 0 1 0 0 0 0 0 1 1 0 0 1 1 1 0 1 0 0 1 0 1 1 1 0 0 0 1 1 1 0; 0 0 0 1 0 0 1 0 0 1 1 0 0 1 1 0 0 0 1 1 1 0 1 0 0 0 1 1 0 0 0 1 1 0 1 1 0 0 0 1 1 1 0 1 1 0 0 1 0 0; 0 0 1 0 0 1 0 1 0 0 1 1 1 0 0 1 1 0 0 0 1 1 0 0 1 0 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 0 1 0 1 1 0 0 1; 1 1 1 0 0 1 0 0 0 0 0 0 0 1 0 1 1 1 1 1 0 1 0 1 1 0 1 1 1 0 1 0 0 0 1 0 1 0 1 0 1 0 0 1 1 0 1 1 0 0; 0 0 1 0 1 0 0 1 0 0 0 1 1 0 1 1 1 0 1 1 1 1 0 1 1 1 0 1 0 0 1 0 1 1 0 1 0 1 0 1 1 0 0 0 0 1 0 1 1 0; 0 1 0 0 0 0 0 0 0 1 1 0 1 1 1 0 0 0 1 1 0 0 1 1 1 0 1 1 0 1 1 0 0 1 0 0 0 1 1 1 0 0 1 1 0 0 1 1 1 0; 0 0 0 1 0 1 1 1 1 1 1 0 0 0 0 0 1 0 0 1 0 0 0 0 1 1 1 1 0 0 0 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 1 1 0 1; 1 0 0 1 0 1 0 1 1 1 1 0 0 1 1 0 0 1 1 0 1 0 1 1 0 1 1 1 1 1 0 1 0 1 0 0 1 1 0 0 0 1 1 1 0 1 1 0 1 1; 0 1 0 0 0 0 0 1 0 1 0 1 0 1 1 1 0 1 1 1 0 1 0 1 0 1 1 0 1 0 1 1 0 1 0 0 1 0 1 0 1 0 0 0 1 1 1 1 1 0; 0 0 1 0 0 1 0 1 0 0 0 0 0 1 0 1 1 1 1 1 1 1 1 0 1 0 0 0 1 1 0 1 0 0 0 0 0 1 1 0 0 0 1 1 0 1 0 0 0 0; 0 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 0 1 0 0 0 0 0 1 1 0 1 1 0 0 1 0 1 0 0 1 1 0 0 1 1 0 0 0 0 0 1; 0 1 0 1 1 0 0 1 1 0 0 0 1 0 1 0 0 0 1 1 1 1 1 0 0 0 0 1 1 0 1 0 1 0 1 1 1 1 1 0 0 1 0 1 1 1 1 1 1 1; 1 1 1 0 1 1 0 0 0 1 1 1 0 0 1 1 0 1 0 1 1 1 1 0 1 0 0 1 0 1 0 0 0 1 1 0 1 1 0 0 0 1 0 1 1 1 1 0 1 0; 1 1 0 0 0 1 0 1 1 0 1 1 0 0 0 1 0 1 0 1 0 0 0 1 1 1 1 0 1 0 0 1 1 0 0 1 1 0 1 1 0 1 1 0 1 1 1 0 1 1; 0 1 1 0 0 1 1 0 0 0 1 1 0 1 0 0 0 1 0 0 0 1 1 1 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 0 1 0 1 1 0 1 0 1; 1 1 0 0 0 1 1 1 1 0 1 1 0 0 0 0 1 1 0 0 0 1 0 1 1 1 1 0 0 1 0 0 1 1 1 1 1 0 0 1 0 0 1 0 1 0 1 0 1 0; 0 1 0 0 0 0 1 1 1 0 0 1 1 1 1 1 0 1 1 1 1 1 1 0 0 1 0 0 1 0 1 0 0 0 1 1 0 0 1 0 1 1 0 1 1 1 0 1 0 1; 0 1 1 1 0 0 0 1 1 0 0 1 0 1 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 1 1 1 1 0 1 0 1 1 1 0 1 1 1 0 0 1 1 1 1 1; 1 1 0 1 1 1 1 1 0 1 1 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 0 1 1 1 1 0 0 1 1 0 1 0 1 0 0 1 0 1 0 1 0 1 0 0; 1 0 1 0 0 1 0 0 0 0 0 1 1 0 1 0 0 1 0 0 1 0 1 0 0 1 0 1 1 1 0 0 1 1 0 0 1 0 1 1 1 0 0 1 1 1 0 1 1 1; 1 1 1 1 1 0 0 1 0 0 0 1 0 1 1 1 1 0 1 0 1 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 1 1 1 1 1 0 0 0 0 1 0 0 1 0; 1 0 0 0 1 0 1 0 1 0 1 1 1 1 0 0 1 0 1 0 1 1 0 1 1 1 1 0 1 0 0 0 0 1 1 1 1 1 0 1 0 0 0 1 1 1 0 0 1 0; 0 1 0 1 0 0 0 0 1 0 1 0 0 0 0 1 1 0 0 1 1 0 1 1 0 1 1 1 0 0 1 0 1 1 0 0 1 0 0 1 0 0 0 1 0 0 0 0 1 0; 0 1 1 0 0 0 1 1 0 1 0 0 1 1 1 0 1 1 0 1 0 0 1 0 1 0 1 1 1 0 1 1 1 0 0 1 0 0 1 1 1 0 1 0 1 0 1 1 1 1; 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 0 1 0 1 0 1 1 1 1 1 0 1 1 1 1 1 0 0 1 0 1 1 1 0 1 0 1 0 0 0 1 1 1 1 1; 1 0 1 1 1 1 1 1 0 1 0 1 1 1 0 1 0 1 0 1 0 1 0 1 0 0 1 1 0 0 0 1 0 1 1 0 0 0 0 1 1 1 0 0 1 0 0 1 1 0; 1 1 1 1 1 1 0 1 1 1 1 0 0 1 0 1 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 1 0 0 1 1 1 1 0 1 0 1 0 1 1 0 1 0 0 0; 1 1 0 1 0 0 0 1 1 1 0 0 0 0 1 1 0 0 1 0 1 1 0 1 0 1 0 1 1 1 0 0 0 0 0 0 1 0 1 0 1 0 1 1 1 0 1 0 0 1; 1 1 1 1 1 0 0 0 1 0 1 1 0 0 0 1 1 0 1 1 1 1 0 1 0 0 1 1 0 0 0 0 1 0 1 0 1 1 1 0 1 0 1 1 0 1 0 0 1 1; 1 1 1 1 0 0 0 0 0 1 1 0 1 1 0 0 1 1 0 0 0 1 1 0 1 1 0 1 1 0 1 1 0 0 0 1 1 0 1 0 0 0 1 0 0 0 0 0 0 1; 1 0 1 0 0 1 0 0 1 1 0 0 1 0 1 1 1 0 1 1 0 0 0 1 0 1 1 0 0 0 0 1 1 0 1 0 0 1 1 1 1 0 1 0 0 0 1 1 0 0; 0 0 1 0 1 1 1 0 1 1 0 1 0 0 1 1 1 0 1 0 0 1 1 0 1 0 0 0 0 1 0 1 0 0 1 1 0 0 0 0 0 0 0 1 1 0 1 1 0 1]
const p = fill(0.02,scenes)
# The ObjVal should be -586.8199999999999 (MIP)
# the PerfInfo lower bound now calculated is -604.5400000000001 which == zpi_c1()
# Q: precise evaluate
# Q_b: benders cut
# Q_SB: s-benders cut

#
import MathOptInterface as MOI
import Gurobi
import LinearAlgebra

# function vv2m(vv) # vector of vector to matrix
#     m = zeros(length(vv[1]),length(vv))
#     for i in eachindex(vv)
#         m[:,i] .= vv[i]
#     end
#     return m
# end

struct Eshat
    z::Vector{Vector{Float64}}
    theta_z::Vector{Float64}
end


function is_2vec_close(v1,v2)
    return LinearAlgebra.norm(v1-v2,Inf) <= 1e-6
end

function is_int(v::Union{Vector{Float64},Float64})
    return LinearAlgebra.norm(v-round.(v),Inf) <= 1e-6
end

function bin_vec_to_1_places(v::Vector{Real})::Vector{Int}
    return findall(x -> x > .5,v)
end

function distill(z::Vector{Vector{Float64}},t::Vector{Float64})
    z_vec,t_vec = deepcopy(z),deepcopy(t)
    n_vec = Vector{Float64}[]
    nt_vec = Float64[]
    rep_vec = Vector{Float64}[]
    rept_vec = Float64[]
    while !isempty(t_vec)
        rep = 0
        for i in 2:length(t_vec)
            if is_2vec_close(z_vec[1],z_vec[i])
                rep = i
                break
            end
        end
        if rep == 0
            push!(n_vec,popfirst!(z_vec))
            push!(nt_vec,popfirst!(t_vec))
        else # currently z_vec[1] == z_vec[rep]
            delInd = t_vec[1] < t_vec[rep] ? rep : 1
            push!(rep_vec,popat!(z_vec,delInd))
            push!(rept_vec,popat!(t_vec,delInd))
        end
    end
    return n_vec,nt_vec #,rep_vec,rept_vec
end

function silent_new_optimizer()
    o = Gurobi.Optimizer(GRB_ENV); # master
    MOI.set(o,MOI.RawOptimizerAttribute("OutputFlag"),0)
    return o
end

function terms_init(l)::Vector{MOI.ScalarAffineTerm{Float64}}
    return [MOI.ScalarAffineTerm(0.,MOI.VariableIndex(0)) for _ in 1:l]
end

function Q(xt,s_ind)::Float64 # precisely evaluate Q_s(x) at x = xt (not necessary int points)
    @assert -1e-6 <= minimum(xt) && maximum(xt) <= 1. + 1e-6 # validity of trial point
    o = silent_new_optimizer() # the s_ind subproblem with trial point xt
    objterms = terms_init(m+n*m) # if no penalty
    x = similar(xt,MOI.VariableIndex) # copy vector of xt
    cpc = similar(xt,MOI.ConstraintIndex{MOI.VariableIndex, MOI.EqualTo{Float64}})
    for j in 1:m
        x[j] = MOI.add_variable(o) # copy variable named x
        cpc[j] = MOI.add_constraint(o,x[j],MOI.EqualTo(xt[j])) # add copy constr immediately
    end
    y0 = similar(q0,MOI.VariableIndex)
    for j in 1:m
        y0[j] = MOI.add_variable(o)
        MOI.add_constraint(o,y0[j],MOI.GreaterThan(0.))
        objterms[0+j] = MOI.ScalarAffineTerm(q0[j],y0[j])
    end
    y = similar(d,MOI.VariableIndex)
    for i in 1:n # row 1:70
        terms = terms_init(m) # constr 2
        for j in 1:m # col 1:30
            y[i,j] = MOI.add_variable(o)
            MOI.add_constraint(o,y[i,j],MOI.ZeroOne())
            objterms[m+m*(i-1)+j] = MOI.ScalarAffineTerm(-d[i,j],y[i,j]) # negative sign! 
            terms[j] = MOI.ScalarAffineTerm(1.,y[i,j]) # constr 2
        end 
        f = MOI.ScalarAffineFunction(terms, 0.) # constr 2
        MOI.add_constraint(o,f,MOI.EqualTo(h[i,s_ind])) # constr 2
    end
    for j in 1:m
        terms = terms_init(n+2)
        for i in 1:n
            terms[i] = MOI.ScalarAffineTerm(d[i,j],y[i,j])
        end
        terms[n+1] = MOI.ScalarAffineTerm(-1.,y0[j])
        terms[n+2] = MOI.ScalarAffineTerm(-u,x[j]) # use of copy variable
        f = MOI.ScalarAffineFunction(terms, 0.)
        MOI.add_constraint(o,f,MOI.LessThan(0.)) # constr 1
    end
    # obj function and SENSE of the s_ind subproblem
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # optimize!
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    return MOI.get(o,MOI.ObjectiveValue())
end

function Q_B(xt,s_ind)::NamedTuple # From Q: 1, relaxing y ∈ Y; 2, output
    @assert -1e-6 <= minimum(xt) && maximum(xt) <= 1. + 1e-6 # validity of trial point
    o = silent_new_optimizer() # the s_ind subproblem with trial point xt
    objterms = terms_init(m+n*m) # if no penalty
    x = similar(xt,MOI.VariableIndex) # copy vector of xt
    cpc = similar(xt,MOI.ConstraintIndex{MOI.VariableIndex, MOI.EqualTo{Float64}})
    for j in 1:m
        x[j] = MOI.add_variable(o) # copy variable named x
        cpc[j] = MOI.add_constraint(o,x[j],MOI.EqualTo(xt[j])) # add copy constr immediately
    end
    y0 = similar(q0,MOI.VariableIndex)
    for j in 1:m
        y0[j] = MOI.add_variable(o)
        MOI.add_constraint(o,y0[j],MOI.GreaterThan(0.))
        objterms[0+j] = MOI.ScalarAffineTerm(q0[j],y0[j])
    end
    y = similar(d,MOI.VariableIndex)
    for i in 1:n # row 1:70
        terms = terms_init(m) # constr 2
        for j in 1:m # col 1:30
            y[i,j] = MOI.add_variable(o)
            MOI.add_constraint(o,y[i,j],MOI.GreaterThan(0.))
            MOI.add_constraint(o,y[i,j],MOI.LessThan(1.))
            objterms[m+m*(i-1)+j] = MOI.ScalarAffineTerm(-d[i,j],y[i,j]) # negative sign! 
            terms[j] = MOI.ScalarAffineTerm(1.,y[i,j]) # constr 2
        end 
        f = MOI.ScalarAffineFunction(terms, 0.) # constr 2
        MOI.add_constraint(o,f,MOI.EqualTo(h[i,s_ind])) # constr 2
    end
    for j in 1:m
        terms = terms_init(n+2)
        for i in 1:n
            terms[i] = MOI.ScalarAffineTerm(d[i,j],y[i,j])
        end
        terms[n+1] = MOI.ScalarAffineTerm(-1.,y0[j])
        terms[n+2] = MOI.ScalarAffineTerm(-u,x[j]) # use of copy variable
        f = MOI.ScalarAffineFunction(terms, 0.)
        MOI.add_constraint(o,f,MOI.LessThan(0.)) # constr 1
    end
    # obj function and SENSE of the s_ind subproblem
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # optimize!
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    obj = MOI.get(o,MOI.ObjectiveValue())
    slope = MOI.get.(o, MOI.ConstraintDual(), cpc) # slope is given by the solver!
    lambda = -slope # lambda is in section 2.3.BDD p2335
    return (l = lambda,o=obj) # o is the RHS of (10), while s is the -λ in (10)
end

function Q_SB(lambda,s_ind)::NamedTuple # precisely evaluate Q_s(x) at x = xt (not necessary int points)
    o = silent_new_optimizer() # the s_ind subproblem with trial point xt
    objterms = terms_init(m+n*m+m) # penalty has m terms
    x = similar(c,MOI.VariableIndex) # copy vector of xt
    for j in 1:m
        x[j] = MOI.add_variable(o) # copy variable named x
        MOI.add_constraint(o,x[j],MOI.ZeroOne()) # 1st-stage constraint add to copy vector, after relaxing copy constr
    end
    y0 = similar(q0,MOI.VariableIndex)
    for j in 1:m
        y0[j] = MOI.add_variable(o)
        MOI.add_constraint(o,y0[j],MOI.GreaterThan(0.))
        objterms[0+j] = MOI.ScalarAffineTerm(q0[j],y0[j])
    end
    y = similar(d,MOI.VariableIndex)
    for i in 1:n # row 1:70
        terms = terms_init(m) # constr 2
        for j in 1:m # col 1:30
            y[i,j] = MOI.add_variable(o)
            MOI.add_constraint(o,y[i,j],MOI.ZeroOne()) # 2nd-stage do NOT relax Int-constr!
            objterms[m+m*(i-1)+j] = MOI.ScalarAffineTerm(-d[i,j],y[i,j]) # negative sign! 
            terms[j] = MOI.ScalarAffineTerm(1.,y[i,j]) # constr 2
        end 
        f = MOI.ScalarAffineFunction(terms, 0.) # constr 2
        MOI.add_constraint(o,f,MOI.EqualTo(h[i,s_ind])) # constr 2
    end
    for j in 1:m
        terms = terms_init(n+2)
        for i in 1:n
            terms[i] = MOI.ScalarAffineTerm(d[i,j],y[i,j])
        end
        terms[n+1] = MOI.ScalarAffineTerm(-1.,y0[j])
        terms[n+2] = MOI.ScalarAffineTerm(-u,x[j]) # use of copy variable
        f = MOI.ScalarAffineFunction(terms, 0.)
        MOI.add_constraint(o,f,MOI.LessThan(0.)) # constr 1
    end
    # penalty in Obj
    for j in 1:m
        objterms[m+n*m+j] = MOI.ScalarAffineTerm(lambda[j],x[j])
    end
    # obj function and SENSE of the s_ind subproblem
    f = MOI.ScalarAffineFunction(objterms,0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # optimize!
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    xval = MOI.get.(o, MOI.VariablePrimal(), x)
    obj = MOI.get(o,MOI.ObjectiveValue()) # the equation above (10)
    return (l=lambda,o=obj)
end

function Q_star(pai,pai0,s_ind)::Float64 # MIP problem (12)
    @assert pai0 >= -1.0e-6
    o = silent_new_optimizer() # the s_ind subproblem with input(pai,pai0)
    objterms = terms_init(m + m+n*m) # 2 stages
    x = similar(pai,MOI.VariableIndex) # 1st-stage decision vector
    for j in 1:m
        x[j] = MOI.add_variable(o) # copy variable named x
        MOI.add_constraint(o,x[j],MOI.ZeroOne()) # 1st-stage constraint (Int included)
        objterms[0+j] = MOI.ScalarAffineTerm(pai[j],x[j])
    end
    y0 = similar(q0,MOI.VariableIndex)
    for j in 1:m
        y0[j] = MOI.add_variable(o)
        MOI.add_constraint(o,y0[j],MOI.GreaterThan(0.))
        objterms[m+j] = MOI.ScalarAffineTerm(pai0 * q0[j],y0[j])
    end
    y = similar(d,MOI.VariableIndex)
    for i in 1:n # row 1:70
        terms = terms_init(m) # constr 2
        for j in 1:m # col 1:30
            y[i,j] = MOI.add_variable(o)
            MOI.add_constraint(o,y[i,j],MOI.ZeroOne())
            objterms[2m + m*(i-1)+j] = MOI.ScalarAffineTerm(pai0 * -d[i,j],y[i,j]) # negative sign! 
            terms[j] = MOI.ScalarAffineTerm(1.,y[i,j]) # constr 2
        end 
        f = MOI.ScalarAffineFunction(terms, 0.) # constr 2
        MOI.add_constraint(o,f,MOI.EqualTo(h[i,s_ind])) # constr 2
    end
    for j in 1:m
        terms = terms_init(n+2)
        for i in 1:n
            terms[i] = MOI.ScalarAffineTerm(d[i,j],y[i,j])
        end
        terms[n+1] = MOI.ScalarAffineTerm(-1.,y0[j])
        terms[n+2] = MOI.ScalarAffineTerm(-u,x[j])
        f = MOI.ScalarAffineFunction(terms, 0.)
        MOI.add_constraint(o,f,MOI.LessThan(0.)) # constr 1
    end
    # obj function and SENSE of the s_ind subproblem
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    return MOI.get(o,MOI.ObjectiveValue())
end

function Q_star_hat(pai,pai0,s_ind,eshat)::Tuple{Float64, Int64} # take finite minimum (18)
    z_vec, value2_vec = eshat.z, pai0 * eshat.theta_z
    value1_vec = similar(value2_vec)
    for i in eachindex(value2_vec)
        value1_vec[i] = pai' * z_vec[i]
    end
    value_vec = value1_vec .+ value2_vec
    return findmin(value_vec)
end

function isnewtrial(p,x)::Bool
    for c in eachcol(p)
        tmp = maximum(abs.(c - x))
        if tmp < 1e-5
            @warn ("This Trial Already Exists!")
            println()
            @warn x
            println()
            @warn ("This Trial Already Exists!")
            return false
        end
    end
    return true
end

function perf_info_per_scene(s_ind)::NamedTuple # the problem following p_s in (14.9)
    o = silent_new_optimizer()
    objterms = terms_init(m + m+n*m) # stage 1 and stage 2
    x = similar(c,MOI.VariableIndex)
    for j in 1:m
        x[j] = MOI.add_variable(o)
        MOI.add_constraint(o, x[j], MOI.ZeroOne()) # x ∈ X
        objterms[0+j] = MOI.ScalarAffineTerm(c[j],x[j])
    end
    y0 = similar(q0,MOI.VariableIndex)
    for j in 1:m
        y0[j] = MOI.add_variable(o)
        MOI.add_constraint(o,y0[j],MOI.GreaterThan(0.))
        objterms[m+j] = MOI.ScalarAffineTerm(q0[j],y0[j])
    end
    y = similar(d,MOI.VariableIndex)
    for i in 1:n # row 1:70
        terms = terms_init(m) # constr 2
        for j in 1:m # col 1:30
            y[i,j] = MOI.add_variable(o)
            MOI.add_constraint(o,y[i,j],MOI.ZeroOne()) # y ∈ Y
            objterms[2m+m*(i-1)+j] = MOI.ScalarAffineTerm(-d[i,j],y[i,j]) # negative sign! 
            terms[j] = MOI.ScalarAffineTerm(1.,y[i,j]) # constr 2
        end 
        f = MOI.ScalarAffineFunction(terms, 0.) # constr 2
        MOI.add_constraint(o,f,MOI.EqualTo(h[i,s_ind])) # constr 2
    end
    for j in 1:m
        terms = terms_init(n+2)
        for i in 1:n
            terms[i] = MOI.ScalarAffineTerm(d[i,j],y[i,j])
        end
        terms[n+1] = MOI.ScalarAffineTerm(-1.,y0[j])
        terms[n+2] = MOI.ScalarAffineTerm(-u,x[j]) # use of copy variable
        f = MOI.ScalarAffineFunction(terms, 0.)
        MOI.add_constraint(o,f,MOI.LessThan(0.)) # constr 1
    end
    # obj function and SENSE of the s_ind subproblem
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # optimize!
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    objval = MOI.get(o,MOI.ObjectiveValue())
    xval = MOI.get.(o,MOI.VariablePrimal(),x)
    objval_2nd_stage = objval - c' * xval
    return (o = objval, z = xval, theta_z = objval_2nd_stage) # (z, theta_z) in (18)
end

function perf_info_initialization(p)::NamedTuple
    @assert sum(p) == 1. # the probability vector
    vector_of_tuples = perf_info_per_scene.(eachindex(p)) # much computations here
    z_pi = p' * getfield.(vector_of_tuples,:o)
    z_vec = getfield.(vector_of_tuples,:z)
    theta_z_vec = getfield.(vector_of_tuples,:theta_z)
    n_vec,nt_vec = distill(z_vec,theta_z_vec)
    return (z_pi = z_pi, eshat = Eshat(n_vec,nt_vec))
end

function zpi_c1()::Float64
    o = silent_new_optimizer()
    objterms = terms_init(m+scenes)
    x = similar(c,MOI.VariableIndex)
    for j in 1:m
        x[j] = MOI.add_variable(o)
        # MOI.add_constraint(o, x[j], MOI.GreaterThan(0.))
        # MOI.add_constraint(o, x[j], MOI.LessThan(1.))
        objterms[0+j] = MOI.ScalarAffineTerm(c[j],x[j])
    end
    theta = similar(p,MOI.VariableIndex);
    for s_ind in 1:scenes
        theta[s_ind] = MOI.add_variable(o)
        objterms[m+s_ind] = MOI.ScalarAffineTerm(p[s_ind],theta[s_ind])
    end
    # add c1 cut
    for s_ind in 1:scenes
        terms = terms_init(m+1)
        for j in 1:m
            terms[j] = MOI.ScalarAffineTerm(c[j],x[j])
        end
        terms[m+1] = MOI.ScalarAffineTerm(1.,theta[s_ind])
        f = MOI.ScalarAffineFunction(terms, 0.)
        MOI.add_constraint(o,f,MOI.GreaterThan(Q_star(c,1.,s_ind))) # constr 1
    end
    # obj function and SENSE of the s_ind subproblem
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # optimize!
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    return MOI.get(o,MOI.ObjectiveValue())
end


const GRB_ENV = Gurobi.Env()
z_pi, eshat = perf_info_initialization(p)




o = silent_new_optimizer()
objterms = terms_init(m+scenes); # without penalization
x = similar(c,MOI.VariableIndex);
for j in 1:m
    x[j] = MOI.add_variable(o)
    MOI.set(o,MOI.VariableName(),x[j],"x[$j]")
    MOI.add_constraint(o, x[j], MOI.GreaterThan(0.))
    MOI.add_constraint(o, x[j], MOI.LessThan(1.))
    # relaxing this to allow fractional trial point    # MOI.add_constraint(o, x[j], MOI.Integer())
    objterms[0+j] = MOI.ScalarAffineTerm(c[j],x[j])
end
theta = similar(p,MOI.VariableIndex);
for s_ind in 1:scenes
    theta[s_ind] = MOI.add_variable(o) # no lowerbound known initially
    MOI.set(o,MOI.VariableName(),theta[s_ind],"θ[$s_ind]")
    objterms[m+s_ind] = MOI.ScalarAffineTerm(p[s_ind],theta[s_ind])
end

# obj function and SENSE of master
let f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
end
MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)

xt = zeros(m); # initial trial point (Zero Trial Point)
tpool = reshape(xt,(m,1)); # trial_point_pool initialization, adding use tpool = [tpool new_x_trial]
cpool = [zeros(m+1,1) for _ in 1:scenes]; # cut_pool depends on each scene
obj_2nd = similar(p); # store the objective value of the 2nd stage, with scenes

for s_ind in 1:scenes
    obj_2nd[s_ind] = Q(xt,s_ind)
end
ub = c' * xt + p' * obj_2nd # ub initialization = 14325.660000000002

for s_ind in 1:scenes
    ret = Q_B(xt,s_ind)
    terms = [MOI.ScalarAffineTerm.(ret.l,x); MOI.ScalarAffineTerm(1.,theta[s_ind])]
    f = MOI.ScalarAffineFunction(terms, 0.)
    cnst = ret.o + ret.l' * xt
    cpool[s_ind] .= [ret.l; cnst] # (initialization) record the cut coefficient, one pool per scene
    MOI.add_constraint(o,f,MOI.GreaterThan(cnst)) # The initial B cut to avoid Unboundness
end

MOI.optimize!(o) # solve master problem
@assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
lb = MOI.get(o,MOI.ObjectiveValue()) # c' * xt + p' * thetat
@assert lb < ub
@info "$lb < $ub"
xt = MOI.get.(o,MOI.VariablePrimal(),x); # This is the 1st Trial Point
thetat = MOI.get.(o,MOI.VariablePrimal(),theta);
#                   ------ Here starts algorithm ------
for trialPointNumber in 1:30000
    if !isnewtrial(tpool,xt)
        @warn "Main loop exit due to no new Trial Point"
        return
    end
    tpool = [tpool xt]
    trial_is_int = is_int(xt)
    if trial_is_int # update ub
        @info "Int Trial x found in trialPointNumber $trialPointNumber"
        for s_ind in 1:scenes
            obj_2nd[s_ind] = Q(xt,s_ind)
        end
        newub = c' * xt + p' * obj_2nd
        ub = newub < ub ? newub : ub
    end
    for s_ind in 1:scenes
        ret = Q_SB(Q_B(xt,s_ind).l,s_ind) # trial_is_int ? Q_B(xt,s_ind) : Q_SB(Q_B(xt,s_ind).l,s_ind)
        cnst = ret.o # o is the RHS at (10)
        if ret.l' * xt + thetat[s_ind] < cnst - 1e-6 # violation
            terms = [MOI.ScalarAffineTerm.(ret.l,x); MOI.ScalarAffineTerm(1.,theta[s_ind])] # λ = -slope
            f = MOI.ScalarAffineFunction(terms, 0.)
            newcol = [ret.l; cnst]
            cpool[s_ind] = [cpool[s_ind] newcol] # record the cut coefficient, one pool per scene
            MOI.add_constraint(o,f,MOI.GreaterThan(cnst)) # The initial B cut to avoid Unboundness
        else
            @warn ("No Violation Occurs!")
            println("trialPointNumber = $trialPointNumber")
            println("s_ind = $s_ind")
            println("This is xt")
            println(xt)
            println("Theta[ind] is $(thetat[s_ind])")
            println("lhs $(ret.l' * xt + thetat[s_ind]) >= rhs $(cnst)")
            @warn ("No Violation Occurs!")
            return
        end
    end
    MOI.optimize!(o) # solve master problem
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    lb = MOI.get(o,MOI.ObjectiveValue()) # c' * xt + p' * thetat
    @assert lb < ub
    @info "At Trial x $trialPointNumber: $lb < $ub"
    xt = MOI.get.(o,MOI.VariablePrimal(),x)
    thetat = MOI.get.(o,MOI.VariablePrimal(),theta)
end

