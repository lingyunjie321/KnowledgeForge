<script setup lang="ts">
import { ref } from 'vue'
import { triggerUpdate, type UpdateResult } from '../api/update'

const filePath = ref('')
const changeType = ref('modified')
const loading = ref(false)
const result = ref<UpdateResult | null>(null)
const errorMsg = ref('')

const options = [
  { value: 'modified', label: '修改' },
  { value: 'created', label: '新增' },
  { value: 'deleted', label: '删除' },
]

async function submit() {
  if (!filePath.value.trim() || loading.value) return
  loading.value = true
  result.value = null
  errorMsg.value = ''
  try {
    result.value = await triggerUpdate(filePath.value.trim(), changeType.value)
  } catch (e) {
    errorMsg.value = (e as Error).message
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div>
    <div class="mb-6">
      <h2 class="text-lg font-medium text-gray-900">CDC 增量更新</h2>
      <p class="text-sm text-gray-500 mt-0.5">
        当前手动触发 created / modified / deleted；modified 无快照时删旧建新，带快照时按 30% diff 选择 chunk 增量或全量重建
      </p>
    </div>

    <div class="bg-white border border-gray-200 rounded-lg p-6 space-y-4">
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-1.5">文件路径</label>
        <input
          v-model="filePath"
          type="text"
          placeholder="如 uploads/spec.md"
          class="w-full px-3 py-2 rounded-lg border border-gray-300 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          @keydown.enter="submit"
        />
        <p class="text-xs text-gray-500 mt-1">相对项目根目录或绝对路径</p>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 mb-1.5">变更类型</label>
        <div class="flex gap-2">
          <button
            v-for="opt in options"
            :key="opt.value"
            @click="changeType = opt.value"
            class="px-4 py-2 text-sm rounded-lg border transition-colors"
            :class="
              changeType === opt.value
                ? 'bg-indigo-600 text-white border-indigo-600'
                : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
            "
          >
            {{ opt.label }}
          </button>
        </div>
      </div>

      <button
        @click="submit"
        :disabled="loading || !filePath.trim()"
        class="px-5 py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
      >
        {{ loading ? '处理中' : '触发更新' }}
      </button>
    </div>

    <div v-if="errorMsg" class="mt-4 p-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
      {{ errorMsg }}
    </div>

    <div v-if="result" class="mt-4 bg-white border border-gray-200 rounded-lg p-5">
      <div class="flex items-center gap-2 mb-3">
        <span
          class="w-2 h-2 rounded-full"
          :class="result.success ? 'bg-emerald-500' : 'bg-red-500'"
        />
        <span class="text-sm font-medium" :class="result.success ? 'text-emerald-700' : 'text-red-700'">
          {{ result.success ? '处理成功' : '处理失败' }}
        </span>
        <span class="text-xs text-gray-500 ml-auto">
          耗时 {{ result.processing_time_ms.toFixed(0) }} ms
        </span>
      </div>
      <div class="grid grid-cols-2 gap-3 text-sm">
        <div class="bg-emerald-50 rounded-lg px-3 py-2">
          <div class="text-xs text-emerald-700">新增向量</div>
          <div class="text-lg font-medium text-emerald-900 mt-0.5">+{{ result.vectors_added }}</div>
        </div>
        <div class="bg-red-50 rounded-lg px-3 py-2">
          <div class="text-xs text-red-700">删除向量</div>
          <div class="text-lg font-medium text-red-900 mt-0.5">-{{ result.vectors_deleted }}</div>
        </div>
        <div class="bg-emerald-50 rounded-lg px-3 py-2">
          <div class="text-xs text-emerald-700">新增实体</div>
          <div class="text-lg font-medium text-emerald-900 mt-0.5">+{{ result.entities_added }}</div>
        </div>
        <div class="bg-emerald-50 rounded-lg px-3 py-2">
          <div class="text-xs text-emerald-700">新增关系</div>
          <div class="text-lg font-medium text-emerald-900 mt-0.5">+{{ result.relations_added }}</div>
        </div>
      </div>
      <div
        v-if="result.diff && Object.keys(result.diff).length"
        class="mt-3 rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-xs text-gray-600"
      >
        <span>diff {{ Math.round((result.diff.change_ratio ?? 0) * 100) }}%</span>
        <span class="ml-2">{{ result.diff.is_major_change ? '全量重建' : 'chunk 增量' }}</span>
        <span class="ml-2">新增 {{ result.diff.added_count ?? 0 }}</span>
        <span class="ml-2">删除 {{ result.diff.removed_count ?? 0 }}</span>
        <span v-if="result.diff.affected_chunks?.length" class="ml-2">
          chunk {{ result.diff.affected_chunks.length }}
        </span>
      </div>
    </div>
  </div>
</template>
