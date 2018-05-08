package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func stepFunction(x float64) int {
	if x > 0 {
		return 1
	} else {
		return 0
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func relu(x float64) float64 {
	return math.Max(0, x)
}

func runBasics() {
	fmt.Println("Step Function:")
	for _, i := range []float64{-5.00, 5.0, 0.1} {
		fmt.Printf("%v -> %v\n", i, stepFunction(i))
	}
	fmt.Println()

	fmt.Println("Sigmoid:")
	for _, i := range []float64{-5.00, 5.0, 0.1} {
		fmt.Printf("%v -> %v\n", i, sigmoid(i))
	}
	fmt.Println()

	fmt.Println("ReLU:")
	for _, i := range []float64{-5.00, 5.0, 0.1} {
		fmt.Printf("%v -> %v\n", i, relu(i))
	}
	fmt.Println()
}

type FirstNetwork struct {
	w1, b1,
	w2, b2,
	w3, b3 *mat.Dense
}

func initNetwork() *FirstNetwork {
	return &FirstNetwork{
		w1: mat.NewDense(2, 3, []float64{
			0.1, 0.3, 0.5,
			0.2, 0.4, 0.6,
		}),
		b1: mat.NewDense(1, 3, []float64{0.1, 0.2, 0.3}),
		w2: mat.NewDense(3, 2, []float64{
			0.1, 0.4,
			0.2, 0.5,
			0.3, 0.6,
		}),
		b2: mat.NewDense(1, 2, []float64{0.1, 0.2}),
		w3: mat.NewDense(2, 2, []float64{
			0.1, 0.3,
			0.2, 0.4,
		}),
		b3: mat.NewDense(1, 2, []float64{0.1, 0.2}),
	}

}

func sigmoidDense(a *mat.Dense) *mat.Dense {
	var z mat.Dense
	z.Apply(func(_, _ int, v float64) float64 { return sigmoid(v) }, a)
	return &z
}

func identityDense(a *mat.Dense) *mat.Dense {
	var z mat.Dense
	z.Apply(func(_, _ int, v float64) float64 { return v }, a)
	return &z
}

func (n *FirstNetwork) forward(x *mat.Dense) *mat.Dense {
	var a1 mat.Dense
	a1.Mul(x, n.w1)
	a1.Add(&a1, n.b1)

	var z1 *mat.Dense
	z1 = sigmoidDense(&a1)

	var a2 mat.Dense
	a2.Mul(z1, n.w2)
	a2.Add(&a2, n.b2)

	var z2 *mat.Dense
	z2 = sigmoidDense(&a2)

	var a3 mat.Dense
	a3.Mul(z2, n.w3)
	a3.Add(&a3, n.b3)

	var y *mat.Dense
	//y = identityDense(&a3)
	y = softMax(&a3)

	return y
}

func runNetwork() {
	x := mat.NewDense(1, 2, []float64{1.0, 0.5})
	n := initNetwork()
	y := n.forward(x)
	fmt.Println(y)
}

func softMax(a *mat.Dense) *mat.Dense {
	y := mat.DenseCopyOf(a)
	c := mat.Max(y)

	y.Apply(func(_, _ int, v float64) float64 {
		//return math.Exp(v)
		return math.Exp(v - c)
	}, y)

	sum := mat.Sum(y)

	y.Apply(func(_, _ int, v float64) float64 {
		return v / sum
	}, y)

	return y
}

func main() {
	//runBasics()
	//runNetwork()

	/*
		a := mat.NewDense(2, 3, []float64{
			0.1, 0.3, 0.5,
			0.2, 0.4, 0.6,
		})
	*/
	a := mat.NewDense(1, 3, []float64{
		//0.3, 2.9, 4.0,
		1010, 1000, 990,
	})
	fmt.Println(softMax(a))
}
