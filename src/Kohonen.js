import { scaleLinear } from 'd3-scale';
import { extent, mean, deviation } from 'd3-array';
import _ from 'lodash/fp';
import { dist, mult, diff, add, normalize, dotProduct, divide } from './vector';
const mld = require('ml-distance');
const math = require('mathjs')

// lodash/fp random has a fixed arity of 2, without the last (and useful) param
const random = _.random.convert({ fixed: false });

// lodash/fp map has an iteratee with a single arg
const mapWithIndex = _.map.convert({ cap: false });

// A basic implementation of Kohonen map

// The main class
class Kohonen {

  // The constructor needs two params :
  // * neurons : an already built neurons grid as an array
  // * data : data set to consider
  // * maxStep : the max step that will be clamped in scaleStepLearningCoef and
  //             scaleStepNeighborhood
  // * maxLearningCoef
  // * minLearningCoef
  // * maxNeighborhood
  // * minNeighborhood
  // * norm: the normalisation type. options are `zscore` or `max` for min/max normalisation
  // * class_method: method of classification, valid options = `somdi`, `hits`
  // * distance: distance measure between neuron options = `manhattan`, defaults to Euclidian
  //
  // each neuron should provide a 2D vector pos,
  // which refer to the grid position
  //
  // You should use an hexagon grid as it is the easier case
  // to deal with neighborhood.
  //
  // You also should normalized your neighborhood in such a way that 2 neighbors
  // got an euclidian distance of 1 between each other.
  constructor(params) {
    if (params) {
      var properties = {
        maxStep: 10000,
        minLearningCoef: .1,
        maxLearningCoef: .4,
        minNeighborhood: .3,
        maxNeighborhood: 1,
        norm: true,
        class_method: 'somdi',
        omega: 1,
        distance: null,
      }

      properties = _.extend(properties, params);
      var neurons = properties.neurons;
      var data = properties.data;
      var labels = properties.labels;
      var labelEnum = properties.labelEnum;
      var maxStep = properties.maxStep;
      var minLearningCoef = properties.minLearningCoef;
      var maxLearningCoef = properties.maxLearningCoef;
      var minNeighborhood = properties.minNeighborhood;
      var maxNeighborhood = properties.maxNeighborhood;
      var norm = properties.norm;
      var class_method = properties.class_method;
      var distance = properties.distance;

      // data vectors should have at least one dimension
      if (!data[0].length) {
        throw new Error('Kohonen constructor: data vectors should have at least one dimension');
      }

        // all vectors should have the same size
        // all vectors values should be number
        for (let ind in data) {
          if (data[ind].length !== data[0].length) {
            throw new Error('Kohonen constructor: all vectors should have the same size');
          }
          const allNum = _.reduce(
            (seed, current) => seed && !isNaN(current) && isFinite(current),
            true,
            data[ind]
            );
          if(!allNum) {
            throw new Error('Kohonen constructor: all vectors should number values');
          }
        }

        this.size = data[0].length;
        this.numNeurons = neurons.length;
        this.labelEnum = labelEnum;
        this.step = 0;
        this.maxStep = maxStep;
        this.norm = norm;
        this.distance = distance;
        this.minLearningCoef = minLearningCoef;
        this.maxLearningCoef = maxLearningCoef;
        this.minNeighborhood = minNeighborhood;
        this.maxNeighborhood = maxNeighborhood;
        this.class_method = class_method;

        this.commonSetup(data, labels);

        // On each neuron, generate a random vector v
        // of <size> dimension
        // each neuron has [{weight: <vector>, somdi: <vector>, pos: <vector>}]
        const randomInitialVectors = this.generateInitialVectors(labels);
        this.neurons = neurons.map(function(neuron, index) {
          var item = randomInitialVectors[index];
          item.pos = neuron.pos;

          return item;
        });
    }
  }

  // method for serializing the class
  export() {
    var out = _.cloneDeep(this);
    delete out._data;
    delete out.scaleStepLearningCoef;
    delete out.scaleStepNeighborhood;
    return out;
  }

  // method for importing previous settings/model
  import(data, labels, props) {
    // populate properties including overwriting neurons
    var keys = Object.keys(props);
    keys.forEach((key)=>{
      this[key] = props[key];
    });

    // seed data and setup learning and neighbourhood functions
    this.commonSetup(data, labels);
  }

  commonSetup(data, labels) {
    // generate scaleStepLearningCoef,
    // as the learning coef decreases with time
    this.scaleStepLearningCoef = scaleLinear()
    .clamp(true)
    .domain([0, this.maxStep])
    .range([this.maxLearningCoef, this.minLearningCoef]);

    // decrease neighborhood with time
    this.scaleStepNeighborhood = scaleLinear()
    .clamp(true)
    .domain([0, this.maxStep])
    .range([this.maxNeighborhood, this.minNeighborhood]);

    this._data = this.seedLabels(data, labels);
    this.somdiLength = _.max(labels) + 1;
    // normalize data
    if (this.norm) {
      this._data.v = this.normalize(this._data.v);
    }
  }

  // bind the labels and create SOMDI vectors
  seedLabels(data, labels) {
    var numClasses = _.max(labels) + 1;
    var out = {v: [], labels: [], somdi: []}
    data.map((item, index) => {
      var somdi = [];

      if (labels) {
        var currentLabel = labels[index];
        var somdi = new Array(numClasses).fill(0);
        somdi[currentLabel] = 1;
      }

      out.v.push(item);
      out.labels.push(currentLabel);
      out.somdi.push(somdi);
    });

    return out;
  }

  normalize(data) {
    data.forEach((item, index)=>{
      data[index] = normalize(item, this.norm);
    });
    return data;
  }

  // learn and return corresponding neurons for the dataset
  learn(log) {
    for (var i=0; i<this.maxStep; i++) {
      this.learnStep();
      if (log) {
        log(this.neurons, this.step);
      }
    }
  }

  // perform single learning step
  learnStep() {
    // pick index for random sample
    var sampleIndex = this.pickDataIndex();
    var sample = this._data.v[sampleIndex];
    var sampleSOMDI = this._data.somdi[sampleIndex];

    // find bmu
    var bmu = this.findBestMatchingUnit(sample);

    // compute current learning coef
    const currentLearningCoef = this.scaleStepLearningCoef(this.step);

    this.neurons.forEach(neuron => {
      // compute neighborhood
      const currentNeighborhood = this.neighborhood(bmu, neuron);
      const scaleFactor = currentNeighborhood * currentLearningCoef;

      // update weights for neuron
      neuron.weight = this.updateStep(neuron.weight, sample, scaleFactor);

      // also update weights of SOMDI
      neuron.somdi = this.updateStep(neuron.somdi, sampleSOMDI, scaleFactor);
    });

    this.step += 1;
    return this.step;
  }

  // LVQ optimisation
  LVQ(log) {
    var self = this;
    // reset number of steps
    self.step = 0;
    for (var i=0; i<this.maxStep; i++) {
      // pick index for random sample
      var sampleIndex = self.pickDataIndex();
      var sample = self._data.v[sampleIndex];
      var label = self._data.labels[sampleIndex];

      // find bmu
      const bmu = self.findBestMatchingUnit(sample);

      // grab the bmu neuron
      const match = self.getNeuron(bmu.pos);

      if (match) {
        // find out what class we think this neuron is
        var criteria = self.maxIndex(match.neuron.somdi);
        if (self.class_method === 'hits') {
          criteria = self.maxIndex(match.neuron.hits);
        }

        // update the weight of the neuron
        self.neurons[match.index].weight = self.lvqUpdate(match.neuron.weight, sample, label, criteria);

        if (self.class_method === 'somdi') {
          // also update SOMDI weights
          var sampleSOMDI = self._data.somdi[sampleIndex];
          self.neurons[match.index].somdi = self.lvqUpdate(match.neuron.somdi, sampleSOMDI, label, criteria);
        }
      }

      self.step += 1;
      if (log) {
        log(self.neurons, self.step);
      }
    }
  }

  LVQ2(log) {
    var self = this;

    function getCandidate(pos, sample) {
      var neuron = self.getNeuron(pos);

      var criteria = self.maxIndex(neuron.neuron.somdi);
      if (self.class_method === 'hits') {
        criteria = self.maxIndex(neuron.neuron.hits);
      }

      var distance = dist(neuron.neuron.weight, sample);

      return {index: neuron.index, label: criteria, distance};
    }

    // reset number of steps
    self.step = 0;
    for (var i=0; i<this.maxStep; i++) {
      // pick index for random sample
      var sampleIndex = self.pickDataIndex();
      var sample = self._data.v[sampleIndex];
      var label = self._data.labels[sampleIndex];

      // get info for bmu
      const bmu = self.findBestMatchingUnit(sample);
      const a = getCandidate(bmu.pos, sample);

      // grab the next best neuron
      const bmu2 = self.findBestMatchingUnit(sample, 1);
      const b = getCandidate(bmu2.pos, sample);

      // check sample is close enough to boundary
      var s = (1 - self.window) / (1 + self.window);

      if (_.min([a.distance / b.distance, b.distance / a.distance]) < s) {
        if (a.label === label) {
          // make more like a, less like b
          self.neurons[a.index].weight = self.lvqUpdate(self.neurons[a.index].weight, sample, 0, 0);
          self.neurons[b.index].weight = self.lvqUpdate(self.neurons[b.index].weight, sample, 0, 1);

          // also update SOMDI weights
          if (self.class_method === 'somdi') {
            var sampleSMDI = self._data.somdi[sampleIndex];
            self.neurons[a.index].somdi = self.lvqUpdate(self.neurons[a.index].somdi, sampleSOMDI, 0, 0);
            self.neurons[b.index].somdi = self.lvqUpdate(self.neurons[b.index].somdi, sampleSOMDI, 0, 1);
          }
        } else if (b.label === label) {
          // make more like b, less like a
          // make more like a, less like b
          self.neurons[b.index].weight = self.lvqUpdate(self.neurons[b.index].weight, sample, 0, 0);
          self.neurons[a.index].weight = self.lvqUpdate(self.neurons[a.index].weight, sample, 0, 1);

          // also update SOMDI weights
          if (self.class_method === 'somdi') {
            var sampleSMDI = self._data.somdi[sampleIndex];
            self.neurons[b.index].somdi = self.lvqUpdate(self.neurons[b.index].somdi, sampleSOMDI, 0, 0);
            self.neurons[a.index].somdi = self.lvqUpdate(self.neurons[a.index].somdi, sampleSOMDI, 0, 1);

          }
        }
      }

      self.step += 1;
      if (log) {
        log(self.neurons, self.step);
      }
    }
  }

  lvqUpdate(weight, sample, label, criteria) {
    var scaleFactor = this.scaleStepLearningCoef(this.step);

    var converge = false;

    if (criteria === label) {
      converge = true;
    }

    var error = diff(weight, sample);
    const delta = mult(error, scaleFactor);

    if (converge) {
      return add(weight, delta);
    } else {
      return diff(delta, weight)
    }
  }

  updateStep(weight, sample, scaleFactor) {
    // compute delta for the current neuron
    var error = diff(weight, sample);
    const delta = mult(error, scaleFactor);
    return add(weight, delta);
  }

  mapping() {
    var positions = [];

    // reset hit counts for all neurons
    this.neurons.forEach((neuron, index)=> {
      this.neurons[index].hits = Array(this.somdiLength).fill(0);
    });

    for (var i=0; i<this._data.v.length; i++) {
      var sample = this._data.v[i];
      var bmu = this.findBestMatchingUnit(sample);

      // increment the hit count of the BMU with the associated class
      var match = this.getNeuron(bmu.pos);
      if (match) {
        var current = match.neuron.hits;
        var somdi = this._data.somdi[i];

        this.neurons[match.index].hits = add(current, somdi);
      }

      // update positions of BMU for sample
      positions.push([bmu.pos, {class: this._data.labels[i]}]);
    }

    return positions;
  }

  // get the neuron and it's index for a given [x, y] position
  getNeuron(pos) {
    var i = this.neurons.findIndex(function(neuron) {
      return neuron.pos === pos;
    });

    if (i < 0) {
      return null;
    }

    return {neuron: this.neurons[i], index: i};
  }

  // generate somdi index for classIndex defined as input.
  // type used to identify neurons that work well for classification
  SOMDI(classIndex, threshold) {
    var _threshold = 0;
    if (threshold) {
      _threshold = threshold;
    }
    var self = this;

    // find neurons with max somdi score associated with classIndex
    var classNeurons = this.neurons.filter(function(neuron) {
      var maxIndex = self.maxIndex(neuron.somdi);
      neuron.sWeight = neuron.somdi[maxIndex];

      return maxIndex === classIndex && neuron.sWeight > _threshold;
    });

    var positions = classNeurons.map(function(neuron) {
      return neuron.pos;
    });

    // multiply weight by somdiWeight & sum over all neurons
    var somdi = new Array(this.neurons[0].weight.length).fill(0);;
    classNeurons.forEach(function(neuron) {
      var current = mult(neuron.weight, neuron.sWeight);
      somdi = add(somdi, current);
    });

    // divide by the number of activated neurons
    somdi = divide(somdi, classNeurons.length);

    return {somdi: somdi, positions: positions};
  }

  // get the classes for each neuron
  neuronClasses(threshold, hits) {
    var _threshold = 0;
    if (threshold) {
      _threshold = threshold;
    }
    var self = this;
    var out = [];

    this.neurons.forEach(function(neuron) {
      var label = null;

      var index = self.maxIndex(neuron.somdi);
      if (neuron.somdi[index] > threshold) {
        label = index;
      }

      if (hits) {
        label = self.maxIndex(neuron.hits);
      }

      out.push({pos: neuron.pos, class: label});
    });

    return out;
  }

  _predict(testData) {
    var self = this;

    // normalise the test data if norm enabled
    if (this.norm) {
      testData = self.normalize(testData);
    }

    var results = [];
    testData.forEach(function(item, index) {
     var bmu = self.findBestMatchingUnit(item);

      // Hit count based classification
      var winningIndex = -1;
      var match = self.getNeuron(bmu.pos);
      if (match) {
        if (self.class_method === 'hits') {
          var hits = match.neuron.hits;
          winningIndex = self.maxIndex(hits);
        } else {
          // SOMDI based calculation of winning neuron
          var winningIndex = self.maxIndex(match.neuron.somdi);
        }
      }

      results.push(winningIndex);
    });

    return results;
  }

  weights() {
    return this.neurons;
  }

  maxIndex(vector) {
    return  vector.indexOf(_.max(vector))
  }

  // pick a random vector among data
  pickDataIndex() {
    return _.random(0, this._data.v.length - 1);
  }

  generateInitialVectors(labels) {
    var output = [];
    for (var i=0; i<this.numNeurons; i++) {
      var vectorLength = this._data.v[0].length;

      var somdi = null;
      var hits = null;

      if (labels) {
        var somdi = Array(this.somdiLength).fill(0).map(()=>Math.random());
        hits = Array(this.somdiLength).fill(0);
      }

      output.push({
        weight: Array(vectorLength).fill(0).map(()=>Math.random()),
        somdi: somdi,
        hits: hits
      });
    }

    return output;
  }

  // Find closer neuron
  findBestMatchingUnit(target, n) {
    var index = 0;

    // allow to check the next best match for LVQ2
    if (n) {
      index = n;
    }

    var getWeight = (neuron) => {
      if (target.length > neuron.weight.length) {
        return neuron.weight.concat(neuron.somdi);
      }

      return neuron.weight;
    }

    if (this.distance === 'manhattan') {
      return _.flow(
        _.orderBy(
          n => mld.distance.manhattan(target, getWeight(n)),
          'asc',
        ),
        _.nth(index)
      )(this.neurons);
    }

    return _.flow(
      _.orderBy(
        n => dist(target, getWeight(n)),
        'asc',
      ),
      _.nth(index)
    )(this.neurons);
  }

  // http://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
  //
  // http://mathworld.wolfram.com/GaussianFunction.html
  //
  // neighborhood function made with a gaussian
  neighborhood(bmu, neuron) {
    const a = 1;
    const sigmaX = 1;
    const sigmaY = 1;

    return a
      * Math.exp(
        -(Math.pow(neuron.pos[0] - bmu.pos[0], 2) / 2 * Math.pow(sigmaX, 2) + Math.pow(neuron.pos[1] - bmu.pos[1], 2) / 2 * Math.pow(sigmaY, 2))
      )
      * this.scaleStepNeighborhood(this.step);
  }
}

export default Kohonen;
