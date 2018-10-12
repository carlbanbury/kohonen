'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _d3Scale = require('d3-scale');

var _d3Array = require('d3-array');

var _fp = require('lodash/fp');

var _fp2 = _interopRequireDefault(_fp);

var _vector = require('./vector');

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var mld = require('ml-distance');
var math = require('mathjs');

// lodash/fp random has a fixed arity of 2, without the last (and useful) param
var random = _fp2.default.random.convert({ fixed: false });

// lodash/fp map has an iteratee with a single arg
var mapWithIndex = _fp2.default.map.convert({ cap: false });

// A basic implementation of Kohonen map

// The main class

var Kohonen = function () {

  // The constructor needs two params :
  // * neurons : an already built neurons grid as an array
  // * data : data set to consider
  // * maxStep : the max step that will be clamped in scaleStepLearningCoef and
  //             scaleStepNeighborhood
  // * maxLearningCoef
  // * minLearningCoef
  // * maxNeighborhood
  // * minNeighborhood
  //
  // each neuron should provide a 2D vector pos,
  // which refer to the grid position
  //
  // You should use an hexagon grid as it is the easier case
  // to deal with neighborhood.
  //
  // You also should normalized your neighborhood in such a way that 2 neighbors
  // got an euclidian distance of 1 between each other.
  function Kohonen(params) {
    _classCallCheck(this, Kohonen);

    if (params) {
      var properties = {
        maxStep: 10000,
        minLearningCoef: .1,
        maxLearningCoef: .4,
        minNeighborhood: .3,
        maxNeighborhood: 1,
        norm: true,
        class_method: 'somdi', // alternative is 'hits', 'supervised'
        distance: null, // alternative = 'corr', manhattan
        _window: 0.3
      };

      properties = _fp2.default.extend(properties, params);
      var neurons = properties.neurons;
      var data = properties.data;
      var labels = properties.labels;
      var maxStep = properties.maxStep;
      var minLearningCoef = properties.minLearningCoef;
      var maxLearningCoef = properties.maxLearningCoef;
      var minNeighborhood = properties.minNeighborhood;
      var maxNeighborhood = properties.maxNeighborhood;
      var norm = properties.norm;
      var class_method = properties.class_method;
      var distance = properties.distance;
      var _window = properties._window;

      // data vectors should have at least one dimension
      if (!data[0].length) {
        throw new Error('Kohonen constructor: data vectors should have at least one dimension');
      }

      // all vectors should have the same size
      // all vectors values should be number
      for (var ind in data) {
        if (data[ind].length !== data[0].length) {
          throw new Error('Kohonen constructor: all vectors should have the same size');
        }
        var allNum = _fp2.default.reduce(function (seed, current) {
          return seed && !isNaN(current) && isFinite(current);
        }, true, data[ind]);
        if (!allNum) {
          throw new Error('Kohonen constructor: all vectors should number values');
        }
      }

      this.size = data[0].length;
      this.numNeurons = neurons.length;
      this.step = 0;
      this.maxStep = maxStep;
      this.norm = norm;
      this.distance = distance;
      this.window = _window;
      this.minLearningCoef = minLearningCoef;
      this.maxLearningCoef = maxLearningCoef;
      this.minNeighborhood = minNeighborhood;
      this.maxNeighborhood = maxNeighborhood;
      this.class_method = class_method;

      this.commonSetup(data, labels);

      // On each neuron, generate a random vector v
      // of <size> dimension
      // each neuron has [{weight: <vector>, somdi: <vector>, pos: <vector>}]
      var randomInitialVectors = this.generateInitialVectors(labels);
      this.neurons = neurons.map(function (neuron, index) {
        var item = randomInitialVectors[index];
        item.pos = neuron.pos;
        item.score = { // use this to rate the performance of each neuron
          correct: 0,
          incorrect: 0
        };

        return item;
      });
    }
  }

  // method for serializing the class


  _createClass(Kohonen, [{
    key: 'export',
    value: function _export() {
      var out = _fp2.default.cloneDeep(this);
      delete out._data;
      delete out.scaleStepLearningCoef;
      delete out.scaleStepNeighborhood;
      return out;
    }

    // method for importing previous settings/model

  }, {
    key: 'import',
    value: function _import(data, labels, props) {
      var _this = this;

      // populate properties including overwriting neurons
      var keys = Object.keys(props);
      keys.forEach(function (key) {
        _this[key] = props[key];
      });

      // seed data and setup learning and neighbourhood functions
      this.commonSetup(data, labels);
    }
  }, {
    key: 'commonSetup',
    value: function commonSetup(data, labels) {
      // generate scaleStepLearningCoef,
      // as the learning coef decreases with time
      this.scaleStepLearningCoef = (0, _d3Scale.scaleLinear)().clamp(true).domain([0, this.maxStep]).range([this.maxLearningCoef, this.minLearningCoef]);

      // decrease neighborhood with time
      this.scaleStepNeighborhood = (0, _d3Scale.scaleLinear)().clamp(true).domain([0, this.maxStep]).range([this.maxNeighborhood, this.minNeighborhood]);

      this._data = this.seedLabels(data, labels);

      // normalize data
      if (this.norm) {
        this._data.v = this.normalize(this._data.v);
      }
    }

    // bind the labels and create SOMDI vectors

  }, {
    key: 'seedLabels',
    value: function seedLabels(data, labels) {
      var numClasses = _fp2.default.max(labels) + 1;
      var out = { v: [], labels: [], somdi: [] };
      data.map(function (item, index) {
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

    // seed a neuron at a set index using the average of the associated class label

  }, {
    key: 'averageSeed',
    value: function averageSeed(index, dataLabel) {
      var vectors = [];
      var self = this;
      this._data.labels.filter(function (label, index) {
        if (label === dataLabel) {
          vectors.push(self._data.v[index]);
        }
      });

      if (vectors.length > 0) {
        var meanVector = math.mean(vectors, 0);
        this.neurons[index].weight = meanVector;
      }
    }
  }, {
    key: 'normalize',
    value: function normalize(data) {
      // TODO: Scale this properly between 0 and 1
      data.forEach(function (item, index) {
        data[index] = (0, _vector.normalize)(item);
      });
      return data;
    }

    // learn and return corresponding neurons for the dataset

  }, {
    key: 'learn',
    value: function learn(log) {
      for (var i = 0; i < this.maxStep; i++) {
        this.learnStep();
        if (log) {
          log(this.neurons, this.step);
        }
      }
    }

    // perform single learning step

  }, {
    key: 'learnStep',
    value: function learnStep() {
      var _this2 = this;

      // pick index for random sample
      var sampleIndex = this.pickDataIndex();
      var sample = this._data.v[sampleIndex];
      var sampleSOMDI = this._data.somdi[sampleIndex];

      // find bmu
      if (this.class_method === 'supervised') {
        console.log('using supervised classifier');
        var bmu = this.findBestMatchingUnit(sample.concat(sampleSOMDI));
      } else {
        var bmu = this.findBestMatchingUnit(sample);
      }

      // compute current learning coef
      var currentLearningCoef = this.scaleStepLearningCoef(this.step);

      this.neurons.forEach(function (neuron) {
        // compute neighborhood
        var currentNeighborhood = _this2.neighborhood(bmu, neuron);
        var scaleFactor = currentNeighborhood * currentLearningCoef;

        // update weights for neuron
        neuron.weight = _this2.updateStep(neuron.weight, sample, scaleFactor);

        // also update weights of SOMDI
        neuron.somdi = _this2.updateStep(neuron.somdi, sampleSOMDI, scaleFactor);
      });

      this.step += 1;
      return this.step;
    }

    // LVQ optimisation

  }, {
    key: 'LVQ',
    value: function LVQ(log) {
      var self = this;
      // reset number of steps
      self.step = 0;
      for (var i = 0; i < this.maxStep; i++) {
        this.learnStep();

        if (log) {
          log(self.neurons, self.step);
        }
      }
    }
  }, {
    key: 'LVQ2',
    value: function LVQ2(log) {
      var self = this;

      function getCandidate(pos, sample) {
        var neuron = self.getNeuron(pos);

        var criteria = self.maxIndex(neuron.neuron.somdi);
        if (self.class_method === 'hits') {
          criteria = self.maxIndex(neuron.neuron.hits);
        }

        var distance = (0, _vector.dist)(neuron.neuron.weight, sample);

        return { index: neuron.index, label: criteria, distance: distance };
      }

      // reset number of steps
      self.step = 0;
      for (var i = 0; i < this.maxStep; i++) {
        // pick index for random sample
        var sampleIndex = self.pickDataIndex();
        var sample = self._data.v[sampleIndex];
        var label = self._data.labels[sampleIndex];

        // get info for bmu
        var bmu = self.findBestMatchingUnit(sample);
        var a = getCandidate(bmu.pos, sample);

        // grab the next best neuron
        var bmu2 = self.findBestMatchingUnit(sample, 1);
        var b = getCandidate(bmu2.pos, sample);

        // check sample is close enough to boundary
        var s = (1 - self.window) / (1 + self.window);

        if (_fp2.default.min([a.distance / b.distance, b.distance / a.distance]) < s) {
          if (a.label === label) {
            // make more like a, less like b
            self.neurons[a.index].weight = self.lvqUpdate(self.neurons[a.index].weight, sample, 0, 0);
            self.neurons[b.index].weight = self.lvqUpdate(self.neurons[b.index].weight, sample, 0, 1);

            // also update SOMDI weights
            if (self.class_method === 'somdi') {
              var sampleSMDI = self._data.somdi[sampleIndex];
              self.neurons[a.index].somdi = self.updateStep(self.neurons[a.index].somdi, sampleSOMDI, 0, 0);
              self.neurons[b.index].somdi = self.updateStep(self.neurons[b.index].somdi, sampleSOMDI, 0, 1);
            }
          } else if (b.label === label) {
            // make more like b, less like a
            // make more like a, less like b
            self.neurons[b.index].weight = self.lvqUpdate(self.neurons[b.index].weight, sample, 0, 0);
            self.neurons[a.index].weight = self.lvqUpdate(self.neurons[a.index].weight, sample, 0, 1);

            // also update SOMDI weights
            if (self.class_method === 'somdi') {
              var sampleSMDI = self._data.somdi[sampleIndex];
              self.neurons[b.index].somdi = self.updateStep(self.neurons[b.index].somdi, sampleSOMDI, 0, 0);
              self.neurons[a.index].somdi = self.updateStep(self.neurons[a.index].somdi, sampleSOMDI, 0, 1);
            }
          }
        }

        self.step += 1;
        if (log) {
          log(self.neurons, self.step);
        }
      }
    }
  }, {
    key: 'lvqUpdate',
    value: function lvqUpdate(weight, sample, label, criteria) {
      var scaleFactor = this.scaleStepLearningCoef(this.step);

      var converge = false;

      if (criteria === label) {
        converge = true;
      }

      var error = (0, _vector.diff)(weight, sample);
      var delta = (0, _vector.mult)(error, scaleFactor);

      if (converge) {
        return (0, _vector.add)(weight, delta);
      } else {
        return (0, _vector.diff)(delta, weight);
      }
    }
  }, {
    key: 'updateStep',
    value: function updateStep(weight, sample, scaleFactor) {
      // compute delta for the current neuron
      var error = (0, _vector.diff)(weight, sample);
      var delta = (0, _vector.mult)(error, scaleFactor);
      return (0, _vector.add)(weight, delta);
    }
  }, {
    key: 'mapping',
    value: function mapping() {
      var positions = [];
      for (var i = 0; i < this._data.v.length; i++) {
        var sample = this._data.v[i];
        var bmu = this.findBestMatchingUnit(sample);

        // increment the hit count of the BMU with the associated class
        var match = this.getNeuron(bmu.pos);
        if (match) {
          var current = match.neuron.hits;
          var somdi = this._data.somdi[i];

          this.neurons[match.index].hits = (0, _vector.add)(current, somdi);
        }

        // update positions of BMU for sample
        positions.push([bmu.pos, { class: this._data.labels[i] }]);
      }

      return positions;
    }

    // get the neuron and it's index for a given [x, y] position

  }, {
    key: 'getNeuron',
    value: function getNeuron(pos) {
      var i = this.neurons.findIndex(function (neuron) {
        return neuron.pos === pos;
      });

      if (i < 0) {
        return null;
      }

      return { neuron: this.neurons[i], index: i };
    }
  }, {
    key: 'setNeuronScore',
    value: function setNeuronScore(pos, correct) {
      var i = this.neurons.findIndex(function (neuron) {
        return neuron.pos === pos;
      });

      if (i < 0) {
        return null;
      }

      if (correct) {
        this.neurons[i].score.correct += 1;
      } else {
        this.neurons[i].score.incorrect += 1;
      }
    }

    // generate somdi index for classIndex defined as input.
    // type used to identify neurons that work well for classification

  }, {
    key: 'SOMDI',
    value: function SOMDI(classIndex, type, threshold) {
      var _threshold = 0;
      if (threshold) {
        _threshold = threshold;
      }
      var self = this;

      // find neurons with max somdi score associated with classIndex
      var classNeurons = this.neurons.filter(function (neuron) {
        var maxIndex = self.maxIndex(neuron.somdi);
        neuron.sWeight = neuron.somdi[maxIndex];

        if (type) {
          if (type === 'positive') {
            return neuron.score.correct > 0 && neuron.score.incorrect === 0;
          }

          if (type === 'negative') {
            return neuron.score.incorrect > 0 && neuron.score.correct === 0;
          }
        }

        return maxIndex === classIndex && neuron.sWeight > _threshold;
      });

      var positions = classNeurons.map(function (neuron) {
        return neuron.pos;
      });

      // multiply weight by somdiWeight & sum over all neurons
      var somdi = new Array(this.neurons[0].weight.length).fill(0);;
      classNeurons.forEach(function (neuron) {
        var current = (0, _vector.mult)(neuron.weight, neuron.sWeight);
        somdi = (0, _vector.add)(somdi, current);
      });

      // divide by the number of activated neurons
      somdi = (0, _vector.divide)(somdi, classNeurons.length);

      return { somdi: somdi, positions: positions };
    }

    // get the classes for each neuron. Currently based on somdi
    // TODO: add support for hit count and neuron performance

  }, {
    key: 'neuronClasses',
    value: function neuronClasses(threshold, hits) {
      var _threshold = 0;
      if (threshold) {
        _threshold = threshold;
      }
      var self = this;
      var out = [];

      this.neurons.forEach(function (neuron) {
        var label = null;

        var index = self.maxIndex(neuron.somdi);
        if (neuron.somdi[index] > threshold) {
          label = index;
        }

        if (hits) {
          label = self.maxIndex(neuron.hits);
        }

        out.push({ pos: neuron.pos, class: label });
      });

      return out;
    }

    // expects an array of test samples and array of labels with corresponding indexes
    // e.g. testData = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]; testLabels = [1, 0, 2]
    // if hits is true, use hit count for classification, else use SOMDI

  }, {
    key: 'predict',
    value: function predict(testData, testLabels, predictIndex, falseIndex) {
      var self = this;

      // normalise the test data if norm enabled
      if (this.norm) {
        testData = self.normalize(testData);
      }

      var results = [];
      testData.forEach(function (item, index) {
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

        // if sample is of type we want to track performance of
        if (testLabels[index] === predictIndex) {
          if (winningIndex === predictIndex) {
            self.setNeuronScore(bmu.pos, true);
          } else if (winningIndex === falseIndex) {
            // record neurons that classify incorrectly
            self.setNeuronScore(bmu.pos, false);
          }
        }

        results.push(winningIndex);
      });

      return results;
    }
  }, {
    key: 'weights',
    value: function weights() {
      return this.neurons;
    }
  }, {
    key: 'maxIndex',
    value: function maxIndex(vector) {
      return vector.indexOf(_fp2.default.max(vector));
    }

    // pick a random vector among data

  }, {
    key: 'pickDataIndex',
    value: function pickDataIndex() {
      return _fp2.default.random(0, this._data.v.length - 1);
    }
  }, {
    key: 'generateInitialVectors',
    value: function generateInitialVectors(labels) {
      var output = [];
      for (var i = 0; i < this.numNeurons; i++) {
        var vectorLength = this._data.v[0].length;

        var somdi = null;
        var hits = null;

        if (labels) {
          var somdiLength = this._data.somdi[0].length;
          var somdi = Array(somdiLength).fill(0).map(function () {
            return Math.random();
          });
          hits = Array(vectorLength).fill(0);
        }

        output.push({
          weight: Array(vectorLength).fill(0).map(function () {
            return Math.random();
          }),
          somdi: somdi,
          hits: hits
        });
      }

      return output;
    }

    // Find closer neuron

  }, {
    key: 'findBestMatchingUnit',
    value: function findBestMatchingUnit(target, n) {
      var index = 0;

      // allow to check the next best match for LVQ2
      if (n) {
        index = n;
      }

      var getWeight = function getWeight(neuron) {
        if (target.length > neuron.weight.length) {
          return neuron.weight.concat(neuron.somdi);
        }

        return neuron.weight;
      };

      if (this.distance === 'manhattan') {
        return _fp2.default.flow(_fp2.default.orderBy(function (n) {
          return mld.distance.manhattan(target, getWeight(n));
        }, 'asc'), _fp2.default.nth(index))(this.neurons);
      }

      if (this.distance === 'corr') {
        return _fp2.default.flow(_fp2.default.orderBy(function (n) {
          return (0, _vector.dotProduct)(target, getWeight(n));
        }, 'desc'), _fp2.default.nth(index))(this.neurons);
      }

      return _fp2.default.flow(_fp2.default.orderBy(function (n) {
        return (0, _vector.dist)(target, getWeight(n));
      }, 'asc'), _fp2.default.nth(index))(this.neurons);
    }

    // http://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    //
    // http://mathworld.wolfram.com/GaussianFunction.html
    //
    // neighborhood function made with a gaussian

  }, {
    key: 'neighborhood',
    value: function neighborhood(bmu, neuron) {
      var a = 1;
      var sigmaX = 1;
      var sigmaY = 1;

      return a * Math.exp(-(Math.pow(neuron.pos[0] - bmu.pos[0], 2) / 2 * Math.pow(sigmaX, 2) + Math.pow(neuron.pos[1] - bmu.pos[1], 2) / 2 * Math.pow(sigmaY, 2))) * this.scaleStepNeighborhood(this.step);
    }

    // The U-Matrix value of a particular node
    // is the average distance between the node's weight vector and that of its closest neighbors.
    // TODO: reimplement this
    // umatrix() {
    //   const roundToTwo = num => +(Math.round(num + "e+2") + "e-2");
    //   const findNeighors = cn => _.filter(
    //     n => roundToTwo(dist(n.pos, cn.pos)) === 1,
    //     this.neurons
    //   );
    //   return _.map(
    //     n => mean(findNeighors(n).map(nb => dist(nb.v, n.v))),
    //     this.neurons
    //   );
    // }


    // TODO: reimplement this
    // hitMap() {
    //   var that = this;
    //   var classMap = [];
    //   var positions = [];
    //   this.hitCount = [];
    //   var that = this;

    //   // loop through all data and match classes to positions
    //   this.data.forEach(function(item) {
    //     var classLabels = item.slice(-that.classPlanes.length);
    //     var bmu = that.findBestMatchingUnit(item);

    //     // store best matching unit and class index
    //     classMap.push([bmu.pos, classLabels]);
    //     positions.push(bmu.pos);
    //   });

    //   // loop through all positions
    //   positions.forEach(function(position) {
    //     // filter and sum class indexes to get hit count
    //     var matches = classMap.filter(function(result) {
    //       return result[0] === position;
    //     });

    //     var hits = new Array(that.classPlanes.length).fill(0);
    //     matches.forEach(function(match) {
    //       hits = add(hits, match[1]);
    //     });

    //     var winner = -1;
    //     var maxCount = _.max(hits);
    //     var guess = hits.indexOf(maxCount);
    //     var check = hits.lastIndexOf(maxCount);

    //     if (guess === check) {
    //       winner = guess;
    //     }

    //     var meta = {hits: hits, winner: winner};
    //     that.hitCount.push([position, meta]);
    //   });
    // }

    // TODO: reimplement this
    // classifyHits(test) {
    //   if (this.norm) {
    //       // make sure we only normalize the data and not the class planes!!!
    //       var classData = test.slice(-this.classPlanes.length);
    //       var testData = test.slice(0,test.length-this.classPlanes.length);

    //       testData = n.normalize(testData, 'max');
    //       test = testData.concat(classData);
    //   }


    //   if (!this.hitCount) {
    //     return null;
    //   }

    //   var bmu = this.findBestMatchingUnit(test);
    //   var match = this.hitCount.filter(function(item) {
    //     return item[0] === bmu.pos;
    //   });

    //   if (match) {
    //     return match[0];
    //   }

    //   return null;
    // }

  }]);

  return Kohonen;
}();

exports.default = Kohonen;