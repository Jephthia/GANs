const template =
`
<div>
  <div class="controls-container">
    <div class="mb-1"><strong>{{name}} - Kernel</strong></div>
    <div class="d-flex">
      <label class="controls-item">
        <span>Step: {{step}}</span>
        <input class="slider" type="range" step="1" :min="minSteps" :max="maxSteps" v-model="step">
      </label>
      <hr class="controls-seperator">
      <label class="controls-item">
        <span>Input: {{currentInput}}</span>
        <input class="slider" type="range" step="1" min="0" :max="inputSize-1" v-model="currentInput">
      </label>
      <hr class="controls-seperator">
      <label class="controls-item">
        <span>Output: {{currentOutput}}</span>
        <input class="slider" type="range" step="1" min="0" :max="outputSize-1" v-model="currentOutput">
      </label>
      <hr class="controls-seperator">
      <label class="controls-item">
        <span>Font Size: {{fontSize}}</span>
        <input class="slider" type="range" step="0.05" min="0.1" max="2.0" v-model="fontSize">
      </label>
      <hr class="controls-seperator">
      <label class="controls-item">
        <span>Round Decimals:</span>
        <div>
          <input type="checkbox" name="round" v-model="round">
          <input type="number" v-model="roundDecimals" min="0" max="20" :disabled="!round">
        </div>
      </label>
    </div>
  </div>
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
    name: String,
    tensorsData: {
      type: Object,
      required: true,
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
    minSteps() {
      const min = Math.min(...this.steps)
      this.step = min
      return min
    },
    maxSteps() {
      return Math.max(...this.steps)
    },
  },
})
