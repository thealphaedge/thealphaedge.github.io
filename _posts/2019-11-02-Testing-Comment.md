---
layout: post
title: Testing Comment
tags: [Test1, Test2]
---

Here are some contents.

{%- if site.just-comments -%}
<div class="just-comments" data-apikey="{{site.just-comments}}"></div>
<script async src="https://just-comments.com/w2.js"></script>
{%- endif -%}
