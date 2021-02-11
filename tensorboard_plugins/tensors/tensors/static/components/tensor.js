const template =
`
<div>
  <div>Tensor Name: {{tensorsData['tag']}}</div>
  <label>
    Font Size:
    <input class="slider" type="range" step="0.05" min="0.1" max="2.0" v-model="fontSize">
  </label>
  <label>
    Step:
    <input class="slider" type="range" step="1" min="0" :max="maxSteps" v-model="step">
    Step {{step}}
  </label>
  <label>
    Round:
    <input type="checkbox" name="round" v-model="round">
  </label>
  <label>
    Decimals
    <input type="number" v-model="roundDecimals" min="0" max="20" :disabled="!round">
  </label>
  <label>
    Input:
    <input class="slider" type="range" step="1" min="0" :max="inputSize-1" v-model="currentInput">
    Current Input: {{currentInput}}
  </label>
  <label>
    Output:
    <input class="slider" type="range" step="1" min="0" :max="outputSize-1" v-model="currentOutput">
    Current Output: {{currentOutput}}
  </label>
  <div class="overflow-auto">
    <div class="tensor-container" :style="tensorContainerStyle">
      <div v-for="row in currentTensor" class="tensor-row">
        <div v-for="col in row" class="tensor-value">
          {{ round && roundDecimals ? col[currentInput][currentOutput].toFixed(roundDecimals) : col[currentInput][currentOutput] }}
        </div>
      </div>
    </div>
  </div>
</div>
`

Vue.component('tensor', {
  template,
  props: {
    tensorsData: {
      type: Object,
      required: true,
      validator: value => value.hasOwnProperty('steps') && value.hasOwnProperty('tag')
    }
  },
  data() {
    return {
      step: 0,
      fontSize: 1,
      round: false,
      roundDecimals: 2,
      currentInput: 0,
      currentOutput: 0,
    }
  },
  computed: {
    tensorContainerStyle() {
      return {
        fontSize: `${this.fontSize}em`
      }
    },
    currentTensor() {
      return this.tensorsData['steps'][this.step.toString()]
    },
    inputSize() {
      return this.currentTensor?.[0]?.[0]?.length || 0
    },
    outputSize() {
      return this.currentTensor?.[0]?.[0]?.[0]?.length || 0
    },
    steps() {
      return Object.keys(this.tensorsData['steps'])
    },
    maxSteps() {
      return Math.max(...this.steps)
    }
  },
})
