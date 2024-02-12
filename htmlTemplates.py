css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    align: right,
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://assets.website-files.com/5d28c3bfa66e813e3a021d7c/5d2e9b58486e5f442e0d3a28_TECHQUANTUM-04_i.jpeg" style="max-width: 140px;border-radius: 25%;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message" style="text-align:right">{{MSG}}</div>
    <div class="avatar">
        <img src="https://i0.wp.com/sitn.hms.harvard.edu/wp-content/uploads/2021/05/human-evolution-gettyimages-122223741.jpeg?resize=768%2C432&ssl=1" style="max-width: 140px;border-radius: 25%;">
    </div>    
    
</div>
'''