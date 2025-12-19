---
layout: default
title: Blog
---

# Blog

{% if site.posts.size > 0 %}
<ul class="post-list">
{% for post in site.posts %}
  <li>
    <span class="post-date">{{ post.date | date: "%B %d, %Y" }}</span>
    <a href="{{ post.url }}">{{ post.title }}</a>
    {% if post.description %}
    <p class="post-description">{{ post.description }}</p>
    {% endif %}
  </li>
{% endfor %}
</ul>
{% else %}
*Coming soon.*
{% endif %}
