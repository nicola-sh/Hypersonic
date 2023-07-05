<?php
use Bitrix\Main\Loader;
use Bitrix\Main\Application;
use Bitrix\Main\Localization\Loc;
use Ipol\AliExpress\Admin\Form;

if (!$USER->IsAdmin()
	|| !Loader::includeModule('ipol.aliexpress')
) {
	return false;
}

Loc::loadMessages(__FILE__);

$form = new Form\Options\Edit();

$request = Application::getInstance()->getContext()->getRequest();
$result  = $form->processRequest($request);

print $form->render($entity, $result);


