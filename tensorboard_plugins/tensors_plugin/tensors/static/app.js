import './components/tensor.js'
import './components/bias.js'

const template =
`
<div>
  <div class="mb-2">
    <strong>Steps</strong> ({{stepsCursor}} - {{stepsCursor+stepsLimit-1}})
    <div>
      <button type="button" @click="previousSteps">Previous</button>
      <button type="button" @click="nextSteps">Next</button>
    </div>
  </div>
  <div v-if="!loading" v-for="tensorsData in allTensorsData" :key="tensorsData.name" class="mb-2">
    <tensor v-if="!noKernel" class='tensor-component' :name="tensorsData.name" :tensorsData="tensorsData['kernel']"></tensor>
    <bias v-if="!noBias" class='bias-component' :name="tensorsData.name" :tensorsData="tensorsData['bias']"></bias>
  </div>
  <div v-if="loading">
    Loading...
  </div>
</div>
`

Vue.component('app', {
  template,
  data() {
    return {
      allTensorsData: [],
      stepsCursor: 0,
      stepsLimit: 5,
      loading: true,
      noKernel: false,
      noBias: true
    }
  },
  methods: {
    async fetchJSON(url, params) {
      const response = params ? await fetch(`${url}?${new URLSearchParams(params)}`) : await fetch(url)
      
      if (!response.ok) return null;
      return response.json();
    },
    previousSteps() {
      if (this.stepsCursor - this.stepsLimit < 0) return
      this.stepsCursor -= this.stepsLimit
      this.loadTensors()
    },
    nextSteps() {
      this.stepsCursor += this.stepsLimit
      this.loadTensors()
    },
    async loadTensors() {
      this.loading = true
      const params = { log_dir: 'testlogs/tensorboard/weights/weights.hdf5', cursor: this.stepsCursor, limit: this.stepsLimit, noKernel: this.noKernel, noBias: this.noBias }
      this.allTensorsData = await this.fetchJSON('./tensors', params)

      console.log('allTensorsData', this.allTensorsData)  
      this.loading = false
    }
  },
  mounted() {
    this.loadTensors()
  }
})
