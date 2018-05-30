package main

import (
	"fmt"
	"math"

	"../common"
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
	WT := this.W.Clone().(*tensor.Dense)
	WT.Transpose()
	xT := this.x.Clone().(*tensor.Dense)
	xT.Transpose()

	dx, _ := dout.MatMul(WT)
	this.dW, _ = xT.MatMul(dout)
	this.db, _ = dout.Sum(0)

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

	tmp, _ := this.y.Sub(this.t)
	dx, _ := tmp.DivScalar(batchSize, true)

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
	ret.layers = []Layer{}

	layer := Layer(NewAffine(ret.W1, ret.b1))

	ret.layers = append(ret.layers, layer)

	/*
			NewAffine(ret.W1, ret.b1).(*Layer),
			&Relu{},
			NewAffine(ret.W2, ret.b2),
		}
	*/
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
	doutInit := 1.0
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

func runSigmoid() {
	dY := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))

	fmt.Println(dY)
	fmt.Println(dY.Sum(0))
	/*
		T1 := tensor.New(tensor.WithBacking(tensor.Range(tensor.Float32, 0, 9)), tensor.WithShape(3, 3))

		d, err := tensor.Exp(T1)
		if err != nil {
			panic(err)
		}
		T1 = d.(*tensor.Dense)
		fmt.Println(T1)

		T1, err = T1.AddScalar(3.0, true)
		if err != nil {
			panic(err)
		}
		fmt.Println(T1)
	*/

	/*
		//var T1, T3, V *tensor.Dense
		var T1, T3 *tensor.Dense
		//var sliced tensor.Tensor
		T1 = tensor.New(tensor.WithBacking(tensor.Range(tensor.Float32, 0, 9)), tensor.WithShape(3, 3))
		T3, _ = T1.PowScalar(float32(-1), true)
		fmt.Printf("Default operation is safe (tensor is left operand)\n==========================\nT3 = T1 ^ 5\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

		//T3, _ = T1.PowScalar(float32(5), false)
		T3, _ = T1.PowScalar(float32(-1), false)
		fmt.Printf("Default operation is safe (tensor is right operand)\n==========================\nT3 = 5 ^ T1\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	*/

	/*
		T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
		sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
		V = sliced.(*tensor.Dense)
		T3, _ = V.PowScalar(float32(5), true)
		fmt.Printf("Default operation is safe (sliced operations - tensor is left operand)\n=============================================\nT3 = T1[0:2, 0:2] ^ 5\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

		T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
		sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
		V = sliced.(*tensor.Dense)
		T3, _ = V.PowScalar(float32(5), false)
		fmt.Printf("Default operation is safe (sliced operations - tensor is right operand)\n=============================================\nT3 = 5 ^ T1[0:2, 0:2]\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)
	*/
}

func main() {
	//runMulLayerFloat()
	//runRelu()
	runSigmoid()
	//runLayers()
}
