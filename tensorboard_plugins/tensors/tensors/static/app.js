import * as Model from './model.js'
import './components/tensor.js'

const template =
`
<div>
  <tensor v-for="tensorsData in allTensorsData" :key="tensorsData.tag" :tensorsData="tensorsData"></tensor>
</div>
`

Vue.component('app', {
  template,
  data() {
    return {
      allTensorsData: [{ steps: {}, tag: '' }]
    }
  },
  async mounted() {
    const res = await Model.getTagsToScalars('.')
    console.log('res', res)

    this.allTensorsData = res
  }
})
