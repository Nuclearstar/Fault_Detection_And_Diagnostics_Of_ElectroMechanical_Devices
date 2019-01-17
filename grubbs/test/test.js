/* eslint func-names:0 */

var grubbs = require('..');
var realWorldTestCases = require('./fixtures/realWorldTestCases');
var assert = require('assert');
var expect = require('chai').expect;
describe('Real world', function () {
  realWorldTestCases.forEach(function (testCase, index) {
    describe(index, function () {
      it('should return the right dataSet', function () {
        var result = grubbs.test(testCase.input);
        assert.deepEqual(
          result[result.length - 1].dataSet.filter(grubbs.isValidData),
          testCase.output
        );
      });
    });
  });
});
