<template>
    <div style="background-color: #f5f6f7;height: 100vh">
        <van-nav-bar style="height: 65px" class="chat-header" :border=false>
            <template #left>

            </template>
            <template #right>
                <van-icon style="transform:rotate(90deg);" name="ellipsis" color="#333333" size="25" />
            </template>
        </van-nav-bar>

        <div class="chat-main" style="">
            <div v-for="(item, index) in msgList" :key="index">
                <div v-if="item.isReceived === false" class="message-item right">
                    {{ item.content }}
                </div>
                <div v-else class="message-item">
                    {{ item.content }}
                </div>
                <div class="item-occupy" />
            </div>
        </div>

        <div class="chat-bar">

            <van-cell-group :border="false" style="margin-top: 8px">
                <van-field v-model="inputMsg" placeholder="Type message..." />
            </van-cell-group>
            <div class="chat-bar-btn send" @click="sendMessage">
                <svg x="1589271304274" class="chat-bar-btn-icon" viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg"
                    p-id="3034" width="30" height="30">
                    <path d="M85.76 896l895.573333-384-895.573333-384-0.426667 298.666667 640 85.333333-640 85.333333z"
                        p-id="3035" fill="#ffffff"></path>
                </svg>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    name: "index",

    data() {
        return {
            inputMsg: '',
            ws: {},
            user: {},
            msgList: [
                {
                    isReceived: true,
                    type: "text",
                    content: '你好呀 !',
                }
            ]
        };
    },

    mounted() {
    },

    methods: {
        async sendMessage() {
            if (this.inputMsg == "" || typeof this.inputMsg === "undefined")
                return;
            const msg = {
                isReceived: false,
                content: this.inputMsg,
                type: "text"
            };
            this.msgList.push(msg);
            this.inputMsg = '';
            const res = await this.$http.post(`/web`, msg);
            this.msgList.push(res.data);
        },

    }
}
</script>

<style scoped>
@import "../../assets/css/chat.css";
</style>