'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _d3Scale = require('d3-scale');

var _d3Array = require('d3-array');

var _fp = require('lodash/fp');

var _fp2 = _interopRequireDefault(_fp);

var _mlPca = require('ml-pca');

var _mlPca2 = _interopRequireDefault(_mlPca);

var _vector = require('./vector');

var _norm = require('norm.js');

var _norm2 = _interopRequireDefault(_norm);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

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
  function Kohonen(_ref) {
    var neurons = _ref.neurons,
        data = _ref.data,
        labels = _ref.labels,
        _ref$maxStep = _ref.maxStep,
        maxStep = _ref$maxStep === undefined ? 10000 : _ref$maxStep,
        _ref$minLearningCoef = _ref.minLearningCoef,
        minLearningCoef = _ref$minLearningCoef === undefined ? .1 : _ref$minLearningCoef,
        _ref$maxLearningCoef = _ref.maxLearningCoef,
        maxLearningCoef = _ref$maxLearningCoef === undefined ? .4 : _ref$maxLearningCoef,
        _ref$minNeighborhood = _ref.minNeighborhood,
        minNeighborhood = _ref$minNeighborhood === undefined ? .3 : _ref$minNeighborhood,
        _ref$maxNeighborhood = _ref.maxNeighborhood,
        maxNeighborhood = _ref$maxNeighborhood === undefined ? 1 : _ref$maxNeighborhood,
        _ref$norm = _ref.norm,
        norm = _ref$norm === undefined ? true : _ref$norm;

    _classCallCheck(this, Kohonen);

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

    // generate scaleStepLearningCoef,
    // as the learning coef decreases with time
    this.scaleStepLearningCoef = (0, _d3Scale.scaleLinear)().clamp(true).domain([0, maxStep]).range([maxLearningCoef, minLearningCoef]);

    // decrease neighborhood with time
    this.scaleStepNeighborhood = (0, _d3Scale.scaleLinear)().clamp(true).domain([0, maxStep]).range([maxNeighborhood, minNeighborhood]);

    this._data = this.seedLabels(data, labels);

    // normalize data
    if (this.norm) {
      this._data.v = this.normalize(this._data.v);
    }

    // On each neuron, generate a random vector v
    // of <size> dimension
    // each neuron has [{weight: <vector>, somdi: <vector>, pos: <vector>}]
    var randomInitialVectors = this.generateInitialVectors(labels);
    this.neurons = neurons.map(function (neuron, index) {
      var item = randomInitialVectors[index];
      item.pos = neuron.pos;
      return item;
    });
  }

  // bind the labels and create SOMDI vectors


  _createClass(Kohonen, [{
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
  }, {
    key: 'normalize',
    value: function normalize(data) {
      // TODO: Scale this properly between 0 and 1
      data.forEach(function (item, index) {
        data[index] = _norm2.default.normalize(item, 'max');
      });
      return data;
    }

    // learn and return corresponding neurons for the dataset

  }, {
    key: 'learn',
    value: function learn(log) {
      var _this = this;

      var self = this;

      var _loop = function _loop() {
        // pick index for random sample
        sampleIndex = _this.pickDataIndex();
        sample = _this._data.v[sampleIndex];

        // find bmu

        var bmu = _this.findBestMatchingUnit(sample);

        // compute current learning coef
        var currentLearningCoef = _this.scaleStepLearningCoef(_this.step);

        _this.neurons.forEach(function (neuron) {
          // compute neighborhood
          var currentNeighborhood = self.neighborhood(bmu, neuron);
          var scaleFactor = currentNeighborhood * currentLearningCoef;

          // update weights for neuron
          neuron.weight = self.updateStep(neuron.weight, sample, scaleFactor);

          // also update weights of SOMDI
          var sampleSOMDI = _this._data.somdi[sampleIndex];
          neuron.somdi = self.updateStep(neuron.somdi, sampleSOMDI, scaleFactor);
        });

        log(_this.neurons, _this.step);
        _this.step += 1;
      };

      for (var i = 0; i < this.maxStep; i++) {
        var sampleIndex;
        var sample;

        _loop();
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
      // TODO: can do hit count in mapping
      // TODO: can also output class
      var positions = [];
      for (var i = 0; i < this._data.v.length; i++) {
        positions.push(this.findBestMatchingUnit(i).pos);
      }

      return positions;
    }

    // expects an array of test samples and array of labels with corresponding indexes
    // e.g. testData = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]; testLabels = [1, 0, 2]

  }, {
    key: 'predict',
    value: function predict(testData, testLabels) {
      var self = this;

      // normalise the test data if norm enabled
      if (this.norm) {
        testData = self.normalize(testData);
      }

      var results = [];
      testData.forEach(function (item, index) {
        var bmu = self.findBestMatchingUnit(item);

        // SOMDI based calculation of winning neuron
        var somdi = bmu.somdi;
        var winningIndex = somdi.indexOf(_fp2.default.max(somdi));
        results.push(winningIndex);
      });

      return results;
    }
  }, {
    key: 'weights',
    value: function weights() {
      return this.neurons;
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

        if (labels) {
          var somdiLength = this._data.somdi[0].length;
          var somdi = Array(somdiLength).fill(0).map(function () {
            return Math.random();
          });
        }

        output.push({
          weight: Array(vectorLength).fill(0).map(function () {
            return Math.random();
          }),
          somdi: somdi
        });
      }

      return output;
    }

    // Find closer neuron

  }, {
    key: 'findBestMatchingUnit',
    value: function findBestMatchingUnit(target) {
      var _neurons = _fp2.default.cloneDeep(this.neurons);

      return _fp2.default.flow(_fp2.default.orderBy(function (n) {
        return (0, _vector.dist)(target, n.weight);
      }, 'asc'), _fp2.default.first)(this.neurons);
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