package main

import (
	"fmt"
	"math"

	"../common"
	"../dataset"
	"gorgonia.org/tensor"
)

type MulLayerFloat struct {
	x float64
	y float64
}

func (this *MulLayerFloat) forward(x, y float64) float64 {
	this.x = x
	this.y = y
	out := x * y

	return out
}

func (this *MulLayerFloat) backward(dout float64) (float64, float64) {
	dx := dout * this.y
	dy := dout * this.x

	return dx, dy
}

func runMulLayerFloat() {
	apple := 100.0
	appleNum := 2.0
	tax := 1.1

	mulAppleLayer := &MulLayerFloat{}
	mulTaxLayer := &MulLayerFloat{}

	applePrice := mulAppleLayer.forward(apple, appleNum)
	price := mulTaxLayer.forward(applePrice, tax)

	fmt.Println(price)

	dprice := 1.0
	dapplePrice, dtax := mulTaxLayer.backward(dprice)
	dapple, dappleNum := mulAppleLayer.backward(dapplePrice)

	fmt.Println(dapple, dappleNum, dtax)
}

type AddLayerFloat struct {
}

func (this *AddLayerFloat) forward(x, y float64) float64 {
	out := x + y
	return out
}

func (this *AddLayerFloat) backward(dout float64) (float64, float64) {
	dx := dout * 1.0
	dy := dout * 1.0
	return dx, dy
}

func runLayers() {
	apple := 100.0
	appleNum := 2.0
	orange := 150.0
	orangeNum := 3.0
	tax := 1.1

	// layer
	mulAppleLayer := &MulLayerFloat{}
	mulOrangeLayer := &MulLayerFloat{}
	addAppleOrangeLayer := &AddLayerFloat{}
	mulTaxLayer := &MulLayerFloat{}

	// forward
	applePrice := mulAppleLayer.forward(apple, appleNum)
	orangePrice := mulOrangeLayer.forward(orange, orangeNum)
	allPrice := addAppleOrangeLayer.forward(applePrice, orangePrice)
	price := mulTaxLayer.forward(allPrice, tax)

	// backward
	dprice := 1.0
	dallPrice, dtax := mulTaxLayer.backward(dprice)
	dapplePrice, dorangePrice := addAppleOrangeLayer.backward(dallPrice)
	dorange, dorangeNum := mulOrangeLayer.backward(dorangePrice)
	dapple, dappleNum := mulAppleLayer.backward(dapplePrice)

	fmt.Println(price)
	fmt.Println(dapple, dappleNum, dorange, dorangeNum, dtax)
}

type Relu struct {
	mask []int
}

func (this *Relu) forward(x *tensor.Dense) *tensor.Dense {
	it := x.Iterator()
	for !it.Done() {
		idx, _ := it.Next()
		a := x.Get(idx).(float64)
		if a < 0 {
			x.Set(idx, 0.0)
			this.mask = append(this.mask, idx)
		}
	}
	return x
}

func (this *Relu) backward(dout *tensor.Dense) *tensor.Dense {
	for _, idx := range this.mask {
		dout.Set(idx, 0.0)
	}
	return dout
}

func runRelu() {
	x := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1.0, -0.5, -2.0, 3.0}))

	relu := &Relu{}

	relu.forward(x)
	fmt.Println(x)
}

type Sigmoid struct {
	out *tensor.Dense
}

func (this *Sigmoid) forward(x *tensor.Dense) *tensor.Dense {
	it := x.Iterator()
	for !it.Done() {
		idx, _ := it.Next()
		a := x.Get(idx).(float64)
		x.Set(idx, 1.0/(1.0+math.Exp(-a)))
	}
	this.out = x
	return this.out
}

func (this *Sigmoid) backward(dout *tensor.Dense) *tensor.Dense {
	it := dout.Iterator()
	for !it.Done() {
		idx, _ := it.Next()
		y := this.out.Get(idx).(float64)
		dd := this.out.Get(idx).(float64)
		dout.Set(idx, dd*(1.0-y)*y)
	}
	return dout
}

type Affine struct {
	W, b, x, dW, db *tensor.Dense
}

func NewAffine(W, b *tensor.Dense) *Affine {
	return &Affine{
		W: W,
		b: b,
	}
}

func (this *Affine) forward(x *tensor.Dense) *tensor.Dense {
	this.x = x
	out, _ := x.MatMul(this.W)
	common.AddVector(out, this.b)
	return out
}

func (this *Affine) backward(dout *tensor.Dense) *tensor.Dense {
	var err error
	WT := this.W.Clone().(*tensor.Dense)
	err = WT.T()
	if err != nil {
		panic(err)
	}
	xT := this.x.Clone().(*tensor.Dense)
	err = xT.T()
	if err != nil {
		panic(err)
	}

	//checkDense(WT, "WT")
	//dx, err := dout.MatMul(WT)
	dx, err := dout.TensorMul(WT, []int{1}, []int{0})
	if err != nil {
		fmt.Println("dout:", dout.Info().Shape())
		fmt.Println("WT:", WT.Info().Shape())
		fmt.Println("W:", this.W.Info().Shape())
		panic(err)
	}
	/*
		checkNaN(xT, "xT")
		fmt.Println(xT)
		checkNaN(dout, "dout")
		fmt.Println(dout)
	*/
	//this.dW, err = xT.MatMul(dout)
	this.dW, err = xT.TensorMul(dout, []int{1}, []int{0})
	if err != nil {
		panic(err)
	}

	//checkDense(this.dW, "dW")
	this.db, err = dout.Sum(0)
	if err != nil {
		panic(err)
	}

	return dx
}

type SoftmaxWithLoss struct {
	loss float64
	y, t *tensor.Dense
}

func (this *SoftmaxWithLoss) forward(x, t *tensor.Dense) float64 {
	this.t = t
	this.y = common.Softmax(x)
	this.loss = common.CrossEntropyError(this.y, this.y)

	return this.loss
}

// dout=1
func (this *SoftmaxWithLoss) backward(dout float64) *tensor.Dense {
	batchSize, _ := this.t.Info().Shape().DimSize(0)

	tmp, err := this.y.Sub(this.t)
	if err != nil {
		panic(err)
	}
	dx, err := tmp.DivScalar(float64(batchSize), true)
	if err != nil {
		panic(err)
	}

	return dx
}

type Layer interface {
	forward(x *tensor.Dense) *tensor.Dense
	backward(dout *tensor.Dense) *tensor.Dense
}

type TwoLayerNet struct {
	W1, b1, W2, b2 *tensor.Dense
	layers         []Layer
	lastLayer      *SoftmaxWithLoss

	affine1Idx, affine2Idx int
}

// wightInitStd = 0.01
func NewTwoLayerNet(inputSize, hiddenSize, outputSize int, weightInitStd float64) *TwoLayerNet {
	ret := &TwoLayerNet{}

	ret.W1 = common.RandomDense(inputSize, hiddenSize)
	ret.b1 = common.RandomDense(hiddenSize)
	ret.W2 = common.RandomDense(hiddenSize, outputSize)
	ret.b2 = common.RandomDense(outputSize)

	ret.W1, _ = ret.W1.MulScalar(weightInitStd, false)
	ret.b1.Zero()
	ret.W2, _ = ret.W2.MulScalar(weightInitStd, false)
	ret.b2.Zero()

	// layers
	ret.layers = []Layer{
		Layer(NewAffine(ret.W1, ret.b1)),
		Layer(&Relu{}),
		Layer(NewAffine(ret.W2, ret.b2)),
	}

	ret.affine1Idx = 0
	ret.affine2Idx = 2

	ret.lastLayer = &SoftmaxWithLoss{}

	return ret
}

func (this *TwoLayerNet) predict(x *tensor.Dense) *tensor.Dense {
	for _, layer := range this.layers {
		x = layer.forward(x)
	}
	return x
}

func (this *TwoLayerNet) loss(x, t *tensor.Dense) float64 {
	y := this.predict(x)
	return this.lastLayer.forward(y, t)
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

func (this *TwoLayerNet) numericalGradient(x, t *tensor.Dense) *TwoLayerNet {
	lossW := func(_ *tensor.Dense) float64 {
		return this.loss(x, t)
	}

	grads := &TwoLayerNet{}

	grads.W1 = common.NumericalGradient(lossW, this.W1)
	grads.b1 = common.NumericalGradient(lossW, this.b1)
	grads.W2 = common.NumericalGradient(lossW, this.W2)
	grads.b2 = common.NumericalGradient(lossW, this.b2)
	return grads
}

func (this *TwoLayerNet) gradient(x, t *tensor.Dense) *TwoLayerNet {
	// forward
	this.loss(x, t)

	// backward
	doutInit := 0.1
	dout := this.lastLayer.backward(doutInit)

	for i := 0; i < len(this.layers); i++ {
		idx := len(this.layers) - 1 - i
		dout = this.layers[idx].backward(dout)
	}

	grads := &TwoLayerNet{}

	grads.W1 = this.layers[this.affine1Idx].(*Affine).dW
	grads.b1 = this.layers[this.affine1Idx].(*Affine).db
	grads.W2 = this.layers[this.affine2Idx].(*Affine).dW
	grads.b2 = this.layers[this.affine2Idx].(*Affine).db
	return grads
}

func NewTensor(data [][]float64) *tensor.Dense {
	x := len(data)
	y := len(data[0])
	flatData := []float64{}
	for _, i := range data {
		flatData = append(flatData, i...)
	}
	return tensor.New(tensor.WithShape(x, y), tensor.WithBacking(flatData))
}

func calDiff(a, b *tensor.Dense) float64 {
	c, _ := a.Sub(b)
	n := 0.0
	sum := 0.0

	it := c.Iterator()
	for !it.Done() {
		idx, _ := it.Next()
		n++
		//fmt.Println(a.Get(idx).(float64), b.Get(idx).(float64), c.Get(idx).(float64))
		sum += math.Abs(c.Get(idx).(float64))
	}
	if sum == 0 {
		return 0
	} else {
		return sum / n
	}
}

func checkDense(a *tensor.Dense, msg ...interface{}) {
	it := a.Iterator()
	for !it.Done() {
		i, _ := it.Next()
		f := a.Get(i).(float64)
		//if f != 0 { fmt.Println(f) }
		if math.IsNaN(f) || math.IsInf(f, 0) {
			fmt.Println("------------------------------")
			fmt.Println(msg...)
			panic("")
		}
	}
}

func numericalGradient() *TwoLayerNet {
	mnist, _ := dataset.LoadMnist(true, true, true)

	network := NewTwoLayerNet(784, 50, 10, 0.01)

	size := 3
	xBatch := NewTensor(mnist.TrainImgNormalized[:size])
	tBatch := NewTensor(common.Byte2Float64Mat(mnist.TrainLabelOneHot[:size]))

	return network.numericalGradient(xBatch, tBatch)
}

func backProp() *TwoLayerNet {
	mnist, _ := dataset.LoadMnist(true, true, true)

	network := NewTwoLayerNet(784, 50, 10, 0.01)

	size := 3
	xBatch := NewTensor(mnist.TrainImgNormalized[:size])
	tBatch := NewTensor(common.Byte2Float64Mat(mnist.TrainLabelOneHot[:size]))

	return network.gradient(xBatch, tBatch)
}

func runGradientCheck() {
	/*
		gradNumerical := numericalGradient()
		gradBackprop := backProp()

		fmt.Println(calDiff(gradNumerical.W1, gradBackprop.W1))
		fmt.Println(calDiff(gradNumerical.b1, gradBackprop.b1))
		fmt.Println(calDiff(gradNumerical.W2, gradBackprop.W2))
		fmt.Println(calDiff(gradNumerical.b2, gradBackprop.b2))
	*/

	mnist, _ := dataset.LoadMnist(true, true, true)

	network := NewTwoLayerNet(784, 50, 10, 0.01)

	size := 3
	xBatch := NewTensor(mnist.TrainImgNormalized[:size])
	tBatch := NewTensor(common.Byte2Float64Mat(mnist.TrainLabelOneHot[:size]))

	gradNumerical := network.numericalGradient(xBatch, tBatch)
	gradBackprop := network.gradient(xBatch, tBatch)

	fmt.Println(calDiff(gradNumerical.W1, gradBackprop.W1))
	fmt.Println(calDiff(gradNumerical.b1, gradBackprop.b1))
	fmt.Println(calDiff(gradNumerical.W2, gradBackprop.W2))
	fmt.Println(calDiff(gradNumerical.b2, gradBackprop.b2))

}

func runTensor() {
	a := NewTensor([][]float64{
		[]float64{11, 12, 13, 14},
		[]float64{21, 22, 23, 24},
	})

	fmt.Println(a)

	b := a.T()
	fmt.Println(a)
	fmt.Println(b)
}

func main() {
	//runMulLayerFloat()
	//runRelu()
	runGradientCheck()
	//runTensor()
}
