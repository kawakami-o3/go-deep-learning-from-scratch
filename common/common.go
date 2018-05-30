package common

import (
	"math"
	"math/rand"

	"gorgonia.org/tensor"
)

func AddVector(dense, vec *tensor.Dense) {
	imax, _ := dense.Info().Shape().DimSize(0)
	jmax, _ := dense.Info().Shape().DimSize(1)

	for i := 0; i < imax; i++ {
		for j := 0; j < jmax; j++ {
			aij, _ := dense.At(i, j)
			vj, _ := vec.At(j)

			dense.SetAt(aij.(float64)+vj.(float64), i, j)
		}
	}
}

func Softmax(a *tensor.Dense) *tensor.Dense {
	imax, _ := a.Info().Shape().DimSize(0)
	jmax, _ := a.Info().Shape().DimSize(1)

	ret := a.Clone().(*tensor.Dense)

	amax, _ := a.Max(1)
	for i := 0; i < imax; i++ {
		s := 0.0
		for j := 0; j < jmax; j++ {
			aij, _ := ret.At(i, j)
			mi, _ := amax.At(i)

			v := math.Exp(aij.(float64) - mi.(float64))
			s += v
			ret.SetAt(v, i, j)
		}

		for j := 0; j < jmax; j++ {
			aij, _ := ret.At(i, j)
			ret.SetAt(aij.(float64)/s, i, j)
		}
	}
	return ret
}

func CrossEntropyError(y, t *tensor.Dense) float64 {
	delta := 1e-7

	yShape := y.Info().Shape()
	batchSize, _ := yShape.DimSize(0)
	cy, _ := yShape.DimSize(1)
	tShape := t.Info().Shape()
	rt, _ := tShape.DimSize(0)
	ct, _ := tShape.DimSize(1)

	if batchSize == rt && cy == ct {
		labels := []float64{}

		argmaxs, _ := t.Argmax(1)
		for i := 0; i < batchSize; i++ {
			a, _ := argmaxs.At(i)
			f := float64(a.(int))
			labels = append(labels, f)
		}

		t = tensor.New(tensor.WithShape(batchSize, 1), tensor.WithBacking(labels))
	}

	sum := 0.0
	for i := 0; i < batchSize; i++ {
		j, _ := t.At(i, 0)
		yij, _ := y.At(i, int(j.(float64)))
		sum += math.Log(yij.(float64) + delta)
	}
	return -sum / float64(batchSize)
}

func NumericalGradient(f func(*tensor.Dense) float64, x *tensor.Dense) *tensor.Dense {
	h := 1e-4

	grad := x.Clone().(*tensor.Dense)

	it := x.Iterator()
	for !it.Done() {
		i, _ := it.Next()

		tmpVal := x.Get(i).(float64)

		x.Set(i, tmpVal+h)
		fxh1 := f(x)

		x.Set(i, tmpVal-h)
		fxh2 := f(x)

		grad.Set(i, (fxh1-fxh2)/(2*h))
		x.Set(i, tmpVal)
	}
	return grad
}

func gaussian() float64 {
	// https://en.wikipedia.org/wiki/Normal_distribution
	x := rand.Float64()
	y := rand.Float64()
	return math.Sqrt(-1*math.Log(x)) * math.Cos(2*math.Pi*y)
}

func RandomDense(dims ...int) *tensor.Dense {
	total := 1
	for _, i := range dims {
		total *= i
	}
	data := []float64{}
	for i := 0; i < total; i++ {
		//data = append(data, float64(total)*rand.Float64())
		data = append(data, gaussian())
		//data = append(data, 1.0) // debug
	}

	return tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
}
