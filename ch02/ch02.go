package main

import (
	"fmt"

	"gonum.org/v1/gonum/blas/blas64"
)

func and(x1, x2 float64) int {
	x := blas64.Vector{Inc: 1, Data: []float64{x1, x2}}
	w := blas64.Vector{Inc: 1, Data: []float64{0.5, 0.5}}
	b := -0.7
	tmp := blas64.Dot(2, w, x) + b
	if tmp <= 0 {
		return 0
	} else {
		return 1
	}
}

func nand(x1, x2 float64) int {
	x := blas64.Vector{Inc: 1, Data: []float64{x1, x2}}
	w := blas64.Vector{Inc: 1, Data: []float64{-0.5, -0.5}}
	b := 0.7
	tmp := blas64.Dot(2, w, x) + b
	if tmp <= 0 {
		return 0
	} else {
		return 1
	}
}

func or(x1, x2 float64) int {
	x := blas64.Vector{Inc: 1, Data: []float64{x1, x2}}
	w := blas64.Vector{Inc: 1, Data: []float64{0.5, 0.5}}
	b := -0.2
	tmp := blas64.Dot(2, w, x) + b
	if tmp <= 0 {
		return 0
	} else {
		return 1
	}
}

func xor(x1, x2 float64) int {
	s1 := float64(nand(x1, x2))
	s2 := float64(or(x1, x2))
	y := and(s1, s2)
	return y
}

func main() {
	tests := [][]float64{
		[]float64{0, 0},
		[]float64{1, 0},
		[]float64{0, 1},
		[]float64{1, 1},
	}
	fmt.Println("AND gate:")
	for _, xs := range tests {
		fmt.Printf("%v -> %v\n", xs, and(xs[0], xs[1]))
	}
	fmt.Println("NAND gate:")
	for _, xs := range tests {
		fmt.Printf("%v -> %v\n", xs, nand(xs[0], xs[1]))
	}
	fmt.Println("OR gate:")
	for _, xs := range tests {
		fmt.Printf("%v -> %v\n", xs, or(xs[0], xs[1]))
	}
	fmt.Println("XOR gate:")
	for _, xs := range tests {
		fmt.Printf("%v -> %v\n", xs, xor(xs[0], xs[1]))
	}
}
