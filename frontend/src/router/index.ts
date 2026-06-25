import { createRouter, createWebHistory } from 'vue-router'
import QA from '../views/QA.vue'
import Graph from '../views/Graph.vue'
import Update from '../views/Update.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/qa' },
    { path: '/qa', name: 'qa', component: QA },
    { path: '/graph', name: 'graph', component: Graph },
    { path: '/update', name: 'update', component: Update },
  ],
})
