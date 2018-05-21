package main

import (
	"fmt"
	"math"

	"gorgonia.org/tensor"

	"gonum.org/v1/gonum/mat"

	"../dataset"
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
	W1, b1, W2, b2 *tensor.Dense
}

func randomDense(dims ...int) *tensor.Dense {
	total := 1
	for _, i := range dims {
		total *= i
	}
	data := []float64{}
	for i := 0; i < total; i++ {
		//data = append(data, float64(total)*rand.Float64())
		data = append(data, 1.0)
	}

	return tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
}

// wightInitStd = 0.01
func NewTwoLayerNet(inputSize, hiddenSize, outputSize int, weightInitStd float64) *TwoLayerNet {
	ret := &TwoLayerNet{}

	ret.W1 = randomDense(inputSize, hiddenSize)
	ret.b1 = randomDense(hiddenSize)
	ret.W2 = randomDense(hiddenSize, outputSize)
	ret.b2 = randomDense(outputSize)

	ret.W1, _ = ret.W1.MulScalar(weightInitStd, false)
	ret.b1.Zero()
	ret.W2, _ = ret.W2.MulScalar(weightInitStd, false)
	ret.b2.Zero()
	return ret
}

func sigmoid(a *tensor.Dense) *tensor.Dense {
	ret := a.Clone().(*tensor.Dense)
	it := ret.Iterator()
	for !it.Done() {
		i, _ := it.Next()
		v := ret.Get(i).(float64)

		ret.Set(i, 1/(1+math.Exp(-v)))
	}
	return ret
}

func max(a *tensor.Dense) float64 {
	i, _ := a.Max()
	return i.ScalarValue().(float64)
}

/*
func sum(a *tensor.Dense) float64 {
	i, _ := a.Sum()
	return i.ScalarValue().(float64)
}
*/

func softmax(a *tensor.Dense) *tensor.Dense {
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

func addVector(dense, vec *tensor.Dense) {

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

func (this *TwoLayerNet) predict(x *tensor.Dense) *tensor.Dense {
	a1, err := x.TensorMul(this.W1, []int{1}, []int{0})
	if err != nil {
		fmt.Println("x:", x.Info().Shape())
		fmt.Println("w1:", this.W1.Info().Shape())
		panic(err)
	}

	addVector(a1, this.b1)
	z1 := sigmoid(a1)
	a2, err := z1.TensorMul(this.W2, []int{1}, []int{0})
	if err != nil {
		fmt.Println("z1:", z1.Info().Shape())
		fmt.Println("w1:", this.W2.Info().Shape())
		panic(err)
	}
	addVector(a2, this.b2)
	y := softmax(a2)

	return y
}

func crossEntropyErrorTensor(y, t *tensor.Dense) float64 {
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

func (this *TwoLayerNet) loss(x, t *tensor.Dense) float64 {
	y := this.predict(x)
	return crossEntropyErrorTensor(y, t)
}

func (this *TwoLayerNet) accuracy(x, t *tensor.Dense) float64 {
	y := this.predict(x)
	ya, _ := y.Argmax(1)
	ta, _ := t.Argmax(1)

	sum := 0
	size := 0
	it := ya.Iterator()
	for !it.Done() {
		i, _ := it.Next()
		yi := ya.Get(i).(float64)
		ti := ta.Get(i).(float64)

		if yi == ti {
			sum++
		}
		size++
	}
	return float64(sum) / float64(size)
}

func numericalGradientTensor(f func(*tensor.Dense) float64, x *tensor.Dense) *tensor.Dense {
	/*
		isize, _ := x.Info().Shape().DimSize(0)
		jsize, _ := x.Info().Shape().DimSize(1)
	*/

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

func (this *TwoLayerNet) numericalGradient(x, t *tensor.Dense) *TwoLayerNet {
	lossW := func(_ *tensor.Dense) float64 {
		return this.loss(x, t)
	}

	grads := &TwoLayerNet{}

	grads.W1 = numericalGradientTensor(lossW, this.W1)
	grads.b1 = numericalGradientTensor(lossW, this.b1)
	grads.W2 = numericalGradientTensor(lossW, this.W2)
	grads.b2 = numericalGradientTensor(lossW, this.b2)
	return grads
}

func (this *TwoLayerNet) Add(network *TwoLayerNet) {
	this.W1, _ = this.W1.Add(network.W1)
	this.b1, _ = this.b1.Add(network.b1)
	this.W2, _ = this.W2.Add(network.W2)
	this.b2, _ = this.b2.Add(network.b2)
}

func (this *TwoLayerNet) MulScalar(n float64) {
	this.W1, _ = this.W1.MulScalar(n, false)
	this.b1, _ = this.b1.MulScalar(n, false)
	this.W2, _ = this.W2.MulScalar(n, false)
	this.b2, _ = this.b2.MulScalar(n, false)
}

func randomBatchMask(limit, size int) []int {
	/*
		store := map[int]int{}
		for len(store) < size {
			i := rand.Intn(limit)
			store[i] = i
		}
		ret := []int{}
		for i, _ := range store {
			ret = append(ret, i)
		}
	*/

	ret := []int{}
	for i := 0; i < size; i++ {
		ret = append(ret, i)
	}
	return ret
}

func byte2Float64(bs []byte) []float64 {
	ret := []float64{}
	for _, b := range bs {
		ret = append(ret, float64(b))
	}
	return ret
}

func runTwoLayerNet() {
	//pp.Println(NewTwoLayerNet(0, 0, 0, 0.01))
	/*
		y := tensor.New(tensor.WithShape(4, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 4, 3, 2, 1}))
		t := tensor.New(tensor.WithShape(4, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 4, 3, 2, 1}))
		fmt.Println(t)
		//fmt.Println(sigmoid(t))
		//fmt.Println(t.At(3, 1))

		pp.Println(crossEntropyErrorTensor(y, t))
	*/

	mnist, _ := dataset.LoadMnist(true, true, true)
	// Hyper Prameters
	itersNum := 10000
	trainSize := len(mnist.TrainImgNormalized)
	batchSize := 100
	learningRate := 0.1

	network := NewTwoLayerNet(784, 50, 10, 0.01)
	//knetwork := NewTwoLayerNet(784, 10, 10, 0.01)
	//network := NewTwoLayerNet(784, 100, 10, 0.01)

	trainLossList := []float64{}
	for i := 0; i < itersNum; i++ {

		batchMask := randomBatchMask(trainSize, batchSize)
		xBatchData := []float64{}
		tBatchData := []float64{}
		for _, idx := range batchMask {
			xBatchData = append(xBatchData, mnist.TrainImgNormalized[idx]...)
			tBatchData = append(tBatchData, byte2Float64(mnist.TrainLabelOneHot[idx])...)
		}
		xBatch := tensor.New(tensor.WithShape(batchSize, dataset.ImgSize), tensor.WithBacking(xBatchData))
		tBatch := tensor.New(tensor.WithShape(batchSize, 10), tensor.WithBacking(tBatchData))

		grad := network.numericalGradient(xBatch, tBatch)

		grad.MulScalar(-learningRate)
		network.Add(grad)

		loss := network.loss(xBatch, tBatch)

		fmt.Println(loss)
		trainLossList = append(trainLossList, loss)
	}

	fmt.Println(trainLossList)
}

func runTensor() {
	/*
		import numpy as np

		a = np.array([[0,1,2],[3,4,5]])
		b = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11]])
		c = a.dot(b)
		print(a)
		print(b)
		print(c)
	*/
	t1 := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking(tensor.Range(tensor.Float64, 0, 6)))
	t2 := tensor.New(tensor.WithShape(3, 4), tensor.WithBacking(tensor.Range(tensor.Float64, 0, 12)))
	//t2 := tensor.New(tensor.WithShape(4, 4), tensor.WithBacking(tensor.Range(tensor.Float64, 0, 16)))
	t3, err := t1.TensorMul(t2, []int{1}, []int{0})
	if err != nil {
		panic(err)
	}
	fmt.Println(t1)
	fmt.Println(t2)
	fmt.Println(t3)
}

func main() {
	//runError()
	//runDiff()
	//runGrad()
	//runNet()
	runTwoLayerNet()

	//runTensor()
}
