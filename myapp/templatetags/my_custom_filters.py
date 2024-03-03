from django import template

register = template.Library()

@register.filter(name='is_instance_of')
def is_instance_of(value, arg):
    """
    Checks if value is an instance of the given class name (arg).
    """
    return isinstance(value, eval(arg))

register = template.Library()

@register.filter(name='hasattr')
def hasattr_filter(value, attr_name):
    """Checks if the given object has an attribute with the specified name."""
    return hasattr(value, attr_name)