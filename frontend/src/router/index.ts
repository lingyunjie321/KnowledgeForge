import { createRouter, createWebHistory } from 'vue-router'
import QA from '../views/QA.vue'
import Graph from '../views/Graph.vue'
import Update from '../views/Update.vue'
import Upload from '../views/Upload.vue'
import Dashboard from '../views/Dashboard.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/qa' },
    { path: '/qa', name: 'qa', component: QA },
    { path: '/graph', name: 'graph', component: Graph },
    { path: '/upload', name: 'upload', component: Upload },
    { path: '/update', name: 'update', component: Update },
    { path: '/dashboard', name: 'dashboard', component: Dashboard },
  ],
})
