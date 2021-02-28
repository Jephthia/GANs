const template =
`
<div>
  <div class="controls-container">
    <div class="mb-1"><strong>{{name}} - Bias</strong></div>
    <div class="d-flex">
      <label class="controls-item">
        <span>Step: {{step}}</span>
        <input class="slider" type="range" step="1" :min="minSteps" :max="maxSteps" v-model="step">
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
    <div class="bias-container" :style="tensorContainerStyle">
      <div v-for="tensor in currentTensor" class="tensor-value">
          {{ round && roundDecimals ? tensor.toFixed(roundDecimals) : tensor }}
      </div>
    </div>
  </div>
</div>
`

Vue.component('bias', {
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
