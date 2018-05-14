package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func meanSquaredError(y, t *mat.Dense) float64 {
	var a mat.Dense
	a.Sub(y, t)
	a.MulElem(&a, &a)
	return 0.5 * mat.Sum(&a)
}

// TODO common
func argMax(m []float64) int {
	vmax := 0.0
	idx := 0
	for i, v := range m {
		if v > vmax {
			vmax = v
			idx = i
		}
	}
	return idx
}

// TODO common
func crossEntropyError(y, t *mat.Dense) float64 {
	delta := 1e-7
	batchSize, cy := y.Dims()
	rt, ct := t.Dims()

	if batchSize == rt && cy == ct {
		var m []float64
		labels := []float64{}

		for i := 0; i < batchSize; i++ {
			labels = append(labels, float64(argMax(mat.Row(m, i, t))))
		}

		t = mat.NewDense(batchSize, 1, labels)
	}

	sum := 0.0
	for i := 0; i < batchSize; i++ {
		j := int(t.At(i, 0))
		sum += math.Log(y.At(i, j) + delta)
	}
	return -sum / float64(batchSize)
}

func runError() {

	t := mat.NewDense(1, 10, []float64{
		0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
	})

	y := mat.NewDense(1, 10, []float64{
		0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0,
	})
	fmt.Println("mean squared error:", meanSquaredError(y, t))
	fmt.Println("cross entropy error:", crossEntropyError(y, t))

	y = mat.NewDense(1, 10, []float64{
		0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0,
	})
	fmt.Println("mean squared error:", meanSquaredError(y, t))
	fmt.Println("cross entropy error:", crossEntropyError(y, t))

}

func numericalDiff(f func(float64) float64, x float64) float64 {
	h := 1e-4
	return (f(x+h) - f(x-h)) / (2 * h)
}

func function1(x float64) float64 {
	return 0.01*math.Pow(x, 2) + 0.1*x
}

func runDiff() {
	fmt.Println(numericalDiff(function1, 5))
	fmt.Println(numericalDiff(function1, 10))
}

func function2(x []float64) float64 {
	return math.Pow(x[0], 2) + math.Pow(x[1], 2)
}

func numericalGradient(f func([]float64) float64, x []float64) []float64 {
	h := 1e-4
	grad := []float64{}

	for idx := 0; idx < len(x); idx++ {
		tmpVal := x[idx]

		x[idx] = tmpVal + h
		fxh1 := f(x)

		x[idx] = tmpVal - h
		fxh2 := f(x)

		grad = append(grad, (fxh1-fxh2)/(2*h))
		x[idx] = tmpVal
	}

	return grad
}

func numericalGradientDense1d(f func(*mat.Dense) float64, x *mat.Dense) *mat.Dense {
	h := 1e-4
	_, c := x.Dims()
	grad := mat.NewDense(1, c, nil)

	for idx := 0; idx < c; idx++ {
		tmpVal := x.At(1, idx)

		x.Set(1, idx, tmpVal+h)
		fxh1 := f(x)

		x.Set(1, idx, tmpVal-h)
		fxh2 := f(x)

		grad.Set(1, idx, (fxh1-fxh2)/(2*h))
		x.Set(1, idx, tmpVal)
	}

	return grad
}

func numericalGradientDense2d(f func(*mat.Dense) float64, X *mat.Dense) *mat.Dense {
	r, c := X.Dims()
	grad := mat.NewDense(r, c, nil)

	for idx := 0; idx < r; idx++ {
		x := make([]float64, c)

		mat.Row(x, idx, X)

		g := numericalGradientDense1d(f, mat.NewDense(1, c, x))

		tmp := make([]float64, c)
		mat.Row(tmp, 1, g)

		grad.SetRow(idx, tmp)
	}

	return grad
}

// lr=0.01, stepNum=100
func gradientDescent(f func([]float64) float64, xInit []float64, lr float64, stepNum int) []float64 {
	x := make([]float64, cap(xInit))
	copy(x, xInit)

	for i := 0; i < stepNum; i++ {
		grad := numericalGradient(f, x)

		for j := 0; j < len(grad); j++ {
			x[j] -= lr * grad[j]
		}
	}

	return x
}

func runGrad() {
	fmt.Println(numericalGradient(function2, []float64{3.0, 4.0}))
	fmt.Println(numericalGradient(function2, []float64{0.0, 2.0}))
	fmt.Println(numericalGradient(function2, []float64{3.0, 0.0}))

	fmt.Println()

	fmt.Println(gradientDescent(function2, []float64{-3.0, 4.0}, 0.1, 100))
	fmt.Println(gradientDescent(function2, []float64{-3.0, 4.0}, 10.0, 100))
	fmt.Println(gradientDescent(function2, []float64{-3.0, 4.0}, 1e-10, 100))
}

type SimpleNet struct {
	W *mat.Dense
}

func NewSimpleNet() *SimpleNet {
	r := 2
	c := 3

	/*
		seed := []float64{}
		for i := 0; i < r*c; i++ {
			seed = append(seed, rand.Float64())
		}
	*/
	seed := []float64{
		0.47355232,
		0.9977393,
		0.84668094,
		0.85557411,
		0.03563661,
		0.69422093,
	}

	return &SimpleNet{
		W: mat.NewDense(r, c, seed),
	}
}

func (this *SimpleNet) Predict(x *mat.Dense) mat.Dense {
	var a mat.Dense
	a.Mul(x, this.W)
	return a
}

// TODO common
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

func (this *SimpleNet) loss(x, t *mat.Dense) float64 {
	z := this.Predict(x)
	y := softMax(&z)
	loss := crossEntropyError(y, t)
	return loss
}

func runNet() {
	net := NewSimpleNet()
	fmt.Println(net.W)

	x := mat.NewDense(1, 2, []float64{0.6, 0.9})
	p := net.Predict(x)

	fmt.Println(p)

	t := mat.NewDense(1, 3, []float64{0, 0, 1})
	loss := net.loss(x, t)

	f := func(_ *mat.Dense) float64 {
		return net.loss(x, t)
	}
	fmt.Println(loss)
	fmt.Println(f)
}

type TwoLayerNet struct {
	W1 *mat.Dense
	b1 *mat.Dense
	W2 *mat.Dense
	b2 *mat.Dense
}

// wightInitStd = 0.01
func NewTwoLayerNet(inputSize, hiddenSize, outputSize int, weightInitStd float64) *TwoLayerNet {
	return &TwoLayerNet{}
}

func main() {
	//runError()
	//runDiff()
	//runGrad()
	//runNet()
}
