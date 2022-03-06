package main

import (
	"bytes"
	"encoding/gob"
	"flag"
	"fmt"
	"image"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"runtime/pprof"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/nfnt/resize"

	"image/color"
	_ "image/jpeg"
	"image/png"
)

const (
	INSERT_TOLERANCE = 3
	MAX_OCTREE_DEPTH = 24
)

type Color struct {
	R, G, B int32
}

type ColorResult struct {
	color     Color
	file_name string
	result    error
}

type Entry struct {
	AvgColor Color
	FileName string
}

type OctreeNode struct {
	Children [8]*OctreeNode
	Elements []Entry
	Center   Color
}

type JobContext struct {
	channelQuit chan bool
	channelJob  chan WorkChunk
	channelJoin chan bool

	src_img            *image.Image
	mosaic_img         *image.RGBA
	root               *OctreeNode
	chunk2mosaic_scale int
	entries            []Entry
	cached_imgs        []*image.Image
	progress           int32
	num_total_chunks   int
}

type WorkChunk struct {
	x0, y0, w, h int
}

var image_dir = flag.String("dir", "", "Directory to scan looking for pictures")
var image_file = flag.String("image", "", "Source image for the mosaic")
var out_file = flag.String("out_file", "mosaic.png", "Output image (the mosaic)")
var chunk_size = flag.Uint("chunk_size", 2, "Chunk size to replace with an individual image. Default=2x2")
var out_chunk_size = flag.Uint("out_chunk_size", 64, "")
var blend_factor = flag.Float64("blend_factor", 0.5, "Blend factor for small mosaic images [0-1], use 0 for single-color blocks")
var match_tolerance = flag.Uint("match_tolerance", 40, "Color tolerance for the image to be considered")
var use_octree = flag.Bool("use_octree", false, "If false, just pick a random image for blocks. Only makes sense with a low blend factor")
var cache_images = flag.Uint("cache_images", 0, "Preload N images and only use these, much faster")
var downscale = flag.Uint("downscale", 16, "How much do we downscale the original image before 'chunking' it (N=scale to 1/Nth of the original)")
var max_color_stdev = flag.Int("max_color_stdev", 40, "If image's color stdev is greater this (for at least 2 channels), we skip it")
var max_matches = flag.Int("max_matches", 6, "Consider up to this many matches for each block. 1=best match only")
var img_ext_flag = flag.String("img_ext", ".jpg", "List of image extension (comma separated)")
var img_ext []string

func (node OctreeNode) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	children_mask := uint8(0)
	for i := 0; i < 8; i++ {
		if node.Children[i] != nil {
			children_mask |= 1 << i
		}
	}
	enc.Encode(children_mask)

	for i := 0; i < 8; i++ {
		if node.Children[i] != nil {
			if err := enc.Encode(*node.Children[i]); err != nil {
				return nil, err
			}
		}
	}

	enc.Encode(node.Elements)
	enc.Encode(node.Center)

	return buf.Bytes(), nil
}
func (node *OctreeNode) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)

	children_mask := uint8(0)
	err := dec.Decode(&children_mask)
	if err != nil {
		fmt.Println(err)
		return err
	}

	for i := 0; i < 8; i++ {
		if children_mask&(1<<i) != 0 {
			node.Children[i] = new(OctreeNode)
			if err = dec.Decode(node.Children[i]); err != nil {
				fmt.Println(err)
				return err
			}
		}
	}

	if err = dec.Decode(&node.Elements); err != nil {
		fmt.Println(err)
		return err
	}
	err = dec.Decode(&node.Center)
	if err != nil {
		fmt.Println(err)
	}
	return err
}

func (node *OctreeNode) insert_node(c *Color, fname *string, depth uint32) {
	child_index := 0
	diffr := c.R - node.Center.R
	diffg := c.G - node.Center.G
	diffb := c.B - node.Center.B
	if diffr > 0 {
		child_index |= 1
	}
	if diffg > 0 {
		child_index |= 2
	}
	if diffb > 0 {
		child_index |= 4
	}
	//fmt.Println("Pushing ", *c, " to ", child_index)

	for i := 0; i < 8; i++ {
		if node.Children[i] == nil {
			node.Children[i] = new(OctreeNode)
		}
	}

	if depth > MAX_OCTREE_DEPTH || (diffr*diffr+diffg*diffg+diffb*diffb) < INSERT_TOLERANCE*INSERT_TOLERANCE {
		e := Entry{AvgColor: *c, FileName: *fname}
		node.Elements = append(node.Elements, e)
	} else {
		if node.Children[child_index] == nil {
			node.Children[child_index] = new(OctreeNode)
		}
		node.Children[child_index].insert_node(c, fname, depth+1)
	}
}

func (node *OctreeNode) find_closest_match(c *Color, tolerance uint32, matches *[]Entry) {
	child_index := 0
	if c.R-node.Center.R > 0 {
		child_index |= 1
	}
	if c.G-node.Center.G > 0 {
		child_index |= 2
	}
	if c.B-node.Center.B > 0 {
		child_index |= 4
	}

	for i := 0; i < len(node.Elements); i++ {
		e := node.Elements[i]
		diffr := c.R - e.AvgColor.R
		diffg := c.G - e.AvgColor.G
		diffb := c.B - e.AvgColor.B
		diff_sq := uint32(diffr*diffr + diffg*diffg + diffb*diffb)

		if diff_sq < tolerance*tolerance {
			*matches = append(*matches, e)
		}
	}

	if node.Children[child_index] == nil {
		return
	}

	node.Children[child_index].find_closest_match(c, tolerance, matches)
}

func load_image(file string) (image.Image, error) {
	reader, err := os.Open(file)

	if err != nil {
		return nil, err
	}
	defer reader.Close()

	m, _, err := image.Decode(reader)
	return m, err
}

func calc_avg_color(file string) ColorResult {

	img, err := load_image(file)
	var result ColorResult
	result.file_name = file
	if err != nil {
		result.result = err
		return result
	}

	smallImage := resize.Resize(32, 32, img, resize.Lanczos3)

	bounds := smallImage.Bounds()
	// Welford's method
	var m [3]int32
	var s [3]int32

	k := uint32(0)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := smallImage.At(x, y).RGBA()
			c := [3]int32{int32(r >> 8), int32(g >> 8), int32(b >> 8)}

			old_m := m
			k++
			for i := range m {
				m[i] += (c[i] - m[i]) / int32(k)
				s[i] += (c[i] - m[i]) * (c[i] - old_m[i])
			}
		}
	}

	k1 := int32(k) - 1
	color_tolerance := *max_color_stdev
	var_tolerance := int32(color_tolerance * color_tolerance)
	cvar := [3]int32{s[0] / k1, s[1] / k1, s[2] / k1}
	num_over := 0
	for i := range cvar {
		if cvar[i] > var_tolerance {
			num_over += 1
		}
	}
	if num_over > 1 {
		result.result = bytes.ErrTooLarge
		return result
	}

	//fmt.Println("Avg color", result.file_name, m)

	result.color.R = m[0]
	result.color.G = m[1]
	result.color.B = m[2]

	return result
}

func calc_avg_color_job(file string, result chan ColorResult) {
	result <- calc_avg_color(file)
}

func arr_contains(arr []string, s string) bool {
	for i := 0; i != len(arr); i++ {
		if arr[i] == s {
			return true
		}
	}
	return false
}

func build_file_list(searchDir string) []Entry {
	fileList := make([]Entry, 0)
	var black_color Color
	e := filepath.Walk(searchDir, func(path string, f os.FileInfo, err error) error {

		if arr_contains(img_ext, strings.ToLower(filepath.Ext(path))) {
			e := Entry{AvgColor: black_color, FileName: path}
			fileList = append(fileList, e)
		}
		//if len(fileList) > 300 {
		//	return filepath.SkipDir
		//}
		return err
	})

	if e != nil {
		panic(e)
	}

	return fileList
}

func build_octree(searchDir string) OctreeNode {

	fileList := build_file_list(searchDir)

	var root OctreeNode

	// Have to do it in batchs, since going super wide can
	// cause us to run out of memory when opening too many images at once.
	BATCH_SIZE := 100
	b := 0

	for {
		batch_end := b + BATCH_SIZE
		if batch_end > len(fileList) {
			batch_end = len(fileList)
		}

		color_channel := make(chan ColorResult, BATCH_SIZE)

		for i := b; i != batch_end; i++ {
			entry := &fileList[i]
			go calc_avg_color_job(entry.FileName, color_channel)
		}
		for i := b; i != batch_end; i++ {
			result := <-color_channel
			if result.result == nil {
				avgColor := result.color
				//fmt.Println("Adding ", result.file_name)
				root.insert_node(&avgColor, &result.file_name, 0)
			}
		}
		close(color_channel)

		fmt.Println("Finished", batch_end, "of", len(fileList), "(", batch_end*100/len(fileList), "%)")
		b = batch_end
		if b == len(fileList) {
			break
		}
	}

	// Test
	//[122 144 160]
	//[88 102 38

	test_color := Color{R: 122, G: 144, B: 160}
	matches := make([]Entry, 0)
	root.find_closest_match(&test_color, 1, &matches)
	fmt.Println("Found", matches[0].FileName, matches[0].AvgColor)

	return root
}

func save_octree(file_name string, root *OctreeNode) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	enc.Encode(root)
	//enc.Encode(root.Children[0]);
	fmt.Println(buf.Len())

	err := ioutil.WriteFile(file_name, buf.Bytes(), 0644)
	if err != nil {
		panic(err)
	}
}

func load_octree(file_name string) OctreeNode {
	buf_bytes, err := ioutil.ReadFile(file_name)
	if err != nil {
		panic(err)
	}
	buf_read := bytes.NewBuffer(buf_bytes)
	tree_decoder := gob.NewDecoder(buf_read)
	var root_in OctreeNode
	err = tree_decoder.Decode(&root_in)
	if err != nil {
		panic(err)
	}
	return root_in
}

func save_image(file string, img image.Image) {
	f, err := os.Create(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	err = png.Encode(f, img)
	if err != nil {
		panic(err)
	}
}

func color_dist(a, b Color) uint32 {
	dr := a.R - b.R
	dg := a.G - b.G
	db := a.B - b.B

	return uint32(dr*dr + dg*dg + db*db)
}

// r,g,b = small image (part of the mosaic)
// c = original image color
func blend_colors(r, g, b uint32, c Color) (uint8, uint8, uint8) {
	t := *blend_factor
	t1 := 1.0 - t
	new_r := float64(r>>8)*t + float64(c.R)*t1
	new_g := float64(g>>8)*t + float64(c.G)*t1
	new_b := float64(b>>8)*t + float64(c.B)*t1
	return uint8(new_r), uint8(new_g), uint8(new_b)
}

func find_good_match(c Color, ctx *JobContext) *Entry {

	if !(*use_octree) {
		idx := rand.Intn(len(ctx.entries))
		return &ctx.entries[idx]
	}

	matches := make([]Entry, 0)
	ctx.root.find_closest_match(&c, uint32(*match_tolerance), &matches)

	if len(matches) > 0 {
		sort.Slice(matches, func(i, j int) bool {
			return (color_dist(matches[i].AvgColor, c) < color_dist(matches[j].AvgColor, c))
		})
		num_matches := len(matches)
		if num_matches > *max_matches {
			num_matches = *max_matches
		}
		idx := rand.Intn(num_matches)
		return &matches[idx]
	}
	return nil
}

func RGBA_from_color(c Color) color.RGBA {
	return color.RGBA{uint8(c.R), uint8(c.G), uint8(c.B), 255}
}

func process_mosaic_chunk(ctx *JobContext, chunk *WorkChunk) {
	end_y := chunk.y0 + chunk.h
	end_x := chunk.x0 + chunk.w
	var total [3]uint32
	for y := chunk.y0; y != end_y; y++ {
		for x := chunk.x0; x != end_x; x++ {
			r, g, b, _ := (*ctx.src_img).At(x, y).RGBA()
			total[0] += r >> 8
			total[1] += g >> 8
			total[2] += b >> 8
		}
	}
	chunk_size := chunk.h * chunk.w
	for k := range total {
		total[k] /= uint32(chunk_size)
	}
	chunk_color := Color{R: int32(total[0]), G: int32(total[1]), B: int32(total[2])}

	block_scale := int(*out_chunk_size)
	block_w := block_scale
	block_h := block_scale
	var block_img image.Image

	if len(ctx.cached_imgs) != 0 {
		cache_index := rand.Intn(len(ctx.cached_imgs))
		block_img = *ctx.cached_imgs[cache_index]
	} else {
		entry := find_good_match(chunk_color, ctx)
		if entry != nil && *blend_factor > 0.0 {
			block_img, _ = load_image(entry.FileName)
			if block_img != nil {
				block_img = resize.Resize(uint(block_w), uint(block_h), block_img, resize.Lanczos3)
			}
		}
	}
	mosaic_img := ctx.mosaic_img

	// x/y are in the "downscaled" coords, we have to convert to dest img coords
	rx := chunk.x0 * ctx.chunk2mosaic_scale
	ry := chunk.y0 * ctx.chunk2mosaic_scale

	dst_offset := mosaic_img.PixOffset(rx, ry)
	dst_line_offset := dst_offset
	for by := 0; by < block_h; by += 1 {
		for bx := 0; bx < block_w; bx += 1 {
			// Small cap improves performance, see https://golang.org/issue/27857
			// (taken from https://cs.opensource.google/go/go/+/refs/tags/go1.17.7:src/image/image.go)
			s := mosaic_img.Pix[dst_offset : dst_offset+4 : dst_offset+4]
			if block_img != nil {
				br, bg, bb, _ := block_img.At(bx, by).RGBA()
				s[0], s[1], s[2] = blend_colors(br, bg, bb, chunk_color)
				s[3] = 0xFF
				//mosaic_img.SetRGBA(rx+bx, ry+by, blend_colors(br, bg, bb, chunk_color))
			} else {
				//mosaic_img.SetRGBA(rx+bx, ry+by, RGBA_from_color(chunk_color))
				s[0] = uint8(chunk_color.R)
				s[1] = uint8(chunk_color.G)
				s[2] = uint8(chunk_color.B)
				s[3] = 0xFF
			}
			dst_offset += 4
		}
		dst_line_offset += mosaic_img.Stride
		dst_offset = dst_line_offset
	}

	progress := atomic.AddInt32(&ctx.progress, 1)
	progress_perc := float32(progress*100) / float32(ctx.num_total_chunks)
	fmt.Println("Progress", progress_perc, "%")
}

func mosaic_worker(ctx *JobContext) {
	channelJob := ctx.channelJob
	for {
		select {
		case chunk := <-channelJob:
			process_mosaic_chunk(ctx, &chunk)
		case <-ctx.channelQuit:
			ctx.channelJoin <- true
			return
		}
	}
}

func prepare_image(work_channel chan string, result chan image.Image) {
	file_name := <-work_channel
	block_img, _ := load_image(file_name)
	block_w := int(*out_chunk_size)
	if block_img != nil {
		block_img = resize.Resize(uint(block_w), uint(block_w), block_img, resize.Lanczos3)
	}
	result <- block_img
}

func build_mosaic() {

	var root OctreeNode
	if *use_octree {
		fmt.Println("Loading octree...")
		root = load_octree("octree.bin")
	}

	original_image, err := load_image(*image_file)
	if err != nil {
		log.Fatal(err)
	}
	original_bounds := original_image.Bounds()
	downscale := *downscale
	rescaled_image := original_image
	if downscale > 1 {
		rescaled_image = resize.Resize(uint(original_bounds.Dx())/downscale, uint(original_bounds.Dy())/downscale,
			original_image, resize.Lanczos3)
	}
	save_image("rescaled.png", rescaled_image)
	bounds := rescaled_image.Bounds()

	rand.Seed(time.Now().UnixNano())

	chunks_w := (bounds.Dx() + int(*chunk_size) - 1) / int(*chunk_size)
	chunks_h := (bounds.Dy() + int(*chunk_size) - 1) / int(*chunk_size)
	out_w := chunks_w * int(*out_chunk_size)
	out_h := chunks_h * int(*out_chunk_size)
	fmt.Println("Output res:", out_w, "x", out_h, "chunks", chunks_w, "x", chunks_h)
	chunk_to_mosaic_scale := *out_chunk_size / *chunk_size

	mosaic_bounds := original_bounds
	mosaic_bounds.Max.X = out_w
	mosaic_bounds.Max.Y = out_h
	mosaic_img := image.NewRGBA(mosaic_bounds)
	fmt.Println("Rescaled bounds: ", bounds)
	fmt.Println("Mosaic bounds: ", mosaic_bounds)

	var context JobContext
	context.root = &root
	context.src_img = &rescaled_image
	context.mosaic_img = mosaic_img
	context.channelJob = make(chan WorkChunk)
	context.channelJoin = make(chan bool)
	context.channelQuit = make(chan bool)

	if !(*use_octree) || *cache_images != 0 {
		start := time.Now()
		context.entries = build_file_list(*image_dir)
		fmt.Println("Building file list took", time.Since(start))

		if *cache_images != 0 {

			start = time.Now()
			// Fisher-Yates
			for i := len(context.entries) - 1; i >= 1; i-- {
				j := rand.Intn(i)
				context.entries[i], context.entries[j] = context.entries[j], context.entries[i]
			}

			work_channel := make(chan string, *cache_images)
			result_channel := make(chan image.Image, *cache_images)
			for i := 0; i != int(*cache_images); i++ {
				go prepare_image(work_channel, result_channel)
				work_channel <- context.entries[i].FileName
			}
			for i := 0; i != int(*cache_images); i++ {
				res := <-result_channel
				context.cached_imgs = append(context.cached_imgs, &res)
				if i%10 == 0 {
					fmt.Println("Cached image: ", i, "/", *cache_images)
				}
			}
			close(work_channel)
			close(result_channel)
			fmt.Println("Caching images took", time.Since(start))
		}
	}

	chunk_size := int(*chunk_size)
	num_total_chunks := (bounds.Dx() * bounds.Dy()) / (chunk_size * chunk_size)
	fmt.Println("Num chunks:", num_total_chunks, chunk_size, "x", chunk_size, "each")

	context.chunk2mosaic_scale = int(chunk_to_mosaic_scale)
	context.num_total_chunks = num_total_chunks

	start_time := time.Now()

	numWorkers := 128
	for w := 0; w < numWorkers; w++ {
		go mosaic_worker(&context)
	}

	for y := bounds.Min.Y; y < bounds.Max.Y; y += chunk_size {
		for x := bounds.Min.X; x < bounds.Max.X; x += chunk_size {

			work_size_x := chunk_size
			work_size_y := chunk_size
			if x+work_size_x > bounds.Max.X {
				work_size_x = bounds.Max.X - x
			}
			if y+work_size_y > bounds.Max.Y {
				work_size_y = bounds.Max.Y - y
			}
			//fmt.Println("Adding job", x, y, x+work_size_x, y+work_size_y)
			context.channelJob <- WorkChunk{x, y, work_size_x, work_size_y}
		}
	}
	for w := 0; w < numWorkers; w++ {
		context.channelQuit <- true
	}
	for w := 0; w < numWorkers; w++ {
		<-context.channelJoin
	}

	fmt.Println("Mosaic took", time.Since(start_time))

	save_image(*out_file, mosaic_img)
}

func print_flag(f *flag.Flag) {
	fmt.Println(f.Name, "-", f.Value)
}

func main() {
	rebuild_tree := flag.Bool("rebuild_octree", false, "Rebuild octree")
	cpuprofile := flag.String("cpuprofile", "", "write cpu profile to file")
	flag.Parse()

	img_ext = strings.Split((*img_ext_flag), ",")

	fmt.Println("Argument values:")
	flag.VisitAll(print_flag)

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	if *rebuild_tree {
		searchDir := *image_dir
		start_time := time.Now()
		root := build_octree(searchDir)
		fmt.Println("Octree construction took", time.Since(start_time))
		save_octree("octree.bin", &root)
	}

	build_mosaic()
}
