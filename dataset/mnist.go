package dataset

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"

	"github.com/k0kubun/pp"
)

const datasetDir = "./"
const saveFile = datasetDir + "mnist.pkl"
const urlBase = "http://yann.lecun.com/exdb/mnist/"

var keyFile = func() map[string]string {
	return map[string]string{
		"train_img":   "train-images-idx3-ubyte.gz",
		"train_label": "train-labels-idx1-ubyte.gz",
		"test_img":    "t10k-images-idx3-ubyte.gz",
		"test_label":  "t10k-labels-idx1-ubyte.gz",
	}
}()

const (
	//trainNum = 60000
	//testNum  = 10000
	imgSize  = 784
	edgeSize = 28
)

type Mnist struct {
	TrainImg   [][]byte
	TrainLabel []byte
	TestImg    [][]byte
	TestLabel  []byte

	TrainImgNormalized [][]float64
	TestImgNormalized  [][]float64

	TrainImg2d           [][][]byte
	TestImg2d            [][][]byte
	TrainImgNormalized2d [][][]float64
	TestImgNormalized2d  [][][]float64

	TrainLabelOneHot [][]byte
	TestLabelOneHot  [][]byte
}

func download(fileName string) error {
	path, err := filepath.Abs(datasetDir + fileName)
	if err != nil {
		return err
	}

	_, err = os.Stat(path)
	if err == nil {
		return nil // the file exists
	}

	resp, err := http.Get(urlBase + fileName)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	ioutil.WriteFile(path, body, os.ModePerm)
	return nil
}

func downloadMnist() error {
	for _, v := range keyFile {
		fmt.Println(v)
		err := download(v)
		if err != nil {
			return err
		}
	}
	return nil
}

func loadLabel(fileName string) ([]byte, error) {
	filePath, err := filepath.Abs(datasetDir + fileName)
	if err != nil {
		return nil, err
	}

	fmt.Println("Converting " + fileName + " to bytes ... ")

	cnt, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	zr, err := gzip.NewReader(bytes.NewBuffer(cnt))
	if err != nil {
		return nil, err
	}

	data, err := ioutil.ReadAll(zr)
	if err != nil {
		return nil, err
	}
	data = data[16:]

	fmt.Println("Done")

	return data, nil
}

func loadImg(fileName string) ([][]byte, error) {
	filePath, err := filepath.Abs(datasetDir + fileName)
	if err != nil {
		return nil, err
	}

	fmt.Println("Converting " + fileName + " to bytes ... ")

	cnt, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	zr, err := gzip.NewReader(bytes.NewBuffer(cnt))
	if err != nil {
		return nil, err
	}

	bytes, err := ioutil.ReadAll(zr)
	if err != nil {
		return nil, err
	}

	bytes = bytes[16:]
	data := [][]byte{}
	for len(bytes) > 0 {
		data = append(data, bytes[0:imgSize])
		bytes = bytes[imgSize:]
	}

	fmt.Println("Done")

	return data, nil
}

func (this *Mnist) convertData() error {
	var err error
	this.TrainImg, err = loadImg(keyFile["train_img"])
	if err != nil {
		return err
	}

	this.TrainLabel, err = loadLabel(keyFile["train_label"])
	if err != nil {
		return err
	}

	this.TestImg, err = loadImg(keyFile["test_img"])
	if err != nil {
		return err
	}

	this.TestLabel, err = loadLabel(keyFile["test_label"])
	if err != nil {
		return err
	}

	return nil
}

func (this *Mnist) init() error {
	err := downloadMnist()
	if err != nil {
		return err
	}

	err = this.convertData()
	if err != nil {
		return err
	}
	return nil
}

func normalize(bs [][]byte) [][]float64 {
	ret := [][]float64{}
	for _, bytes := range bs {
		floats := []float64{}
		for _, b := range bytes {
			floats = append(floats, float64(b)/255.0)
		}
		ret = append(ret, floats)
	}
	return ret
}

func (this *Mnist) normalize() {
	this.TrainImgNormalized = normalize(this.TrainImg)
	this.TestImgNormalized = normalize(this.TestImg)
}

func changeOneHotLabel(bs []byte) [][]byte {
	ret := [][]byte{}
	for _, b := range bs {
		row := []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
		row[b] = 1
		ret = append(ret, row)
	}
	return ret
}

func (this *Mnist) changeOneHotLabel() {
	this.TrainLabelOneHot = changeOneHotLabel(this.TrainLabel)
	this.TestLabelOneHot = changeOneHotLabel(this.TestLabel)
}

func reshapeBytes(bs [][]byte) [][][]byte {
	ret := [][][]byte{}
	for _, bytes := range bs {
		img := [][]byte{}
		for len(bytes) > 0 {
			img = append(img, bytes[0:edgeSize])
			bytes = bytes[edgeSize:]
		}
		ret = append(ret, img)
	}
	return ret
}

func reshapeFloats(fs [][]float64) [][][]float64 {
	ret := [][][]float64{}
	for _, floats := range fs {
		img := [][]float64{}
		for len(floats) > 0 {
			img = append(img, floats[0:edgeSize])
			floats = floats[edgeSize:]
		}
		ret = append(ret, img)
	}
	return ret
}

func (this *Mnist) reshapeImg() {
	if this.TrainImgNormalized == nil {
		this.TrainImg2d = reshapeBytes(this.TrainImg)
		this.TestImg2d = reshapeBytes(this.TestImg)
	} else {
		this.TrainImgNormalized2d = reshapeFloats(this.TrainImgNormalized)
		this.TestImgNormalized2d = reshapeFloats(this.TestImgNormalized)
	}
}

func LoadMnist(normalize, flatten, oneHotLabel bool) (*Mnist, error) {
	mnist := &Mnist{}

	err := mnist.init()
	if err != nil {
		return nil, err
	}

	if normalize {
		mnist.normalize()
	}

	if oneHotLabel {
		mnist.changeOneHotLabel()
	}

	if !flatten {
		mnist.reshapeImg()
	}
	return mnist, nil
}
